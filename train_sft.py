#!/usr/bin/env python3
"""Supervised fine-tuning with LoRA."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer

from data_loader import load_text_dataset

try:
    import yaml
except ImportError:
    yaml = None


class ConfigError(ValueError):
    """Raised when the training configuration is invalid."""


@dataclass
class DatasetConfig:
    name_or_path: Optional[str] = None
    split: str = "train"

    def validate(self) -> None:
        if self.name_or_path is None:
            raise ConfigError("Specify a dataset via `dataset.name_or_path`.")


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[Union[str, List[str]]] = None


@dataclass
class TrainingConfig:
    output_dir: str
    model_name_or_path: str
    seed: int = 42
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_train_epochs: Optional[float] = None
    max_steps: Optional[int] = None
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: Optional[int] = None
    gradient_checkpointing: bool = True
    log_with_tensorboard: bool = True
    logging_dir: Optional[str] = None
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    adam_beta1: Optional[float] = None
    adam_beta2: Optional[float] = None
    adam_epsilon: Optional[float] = None
    resume_from_checkpoint: Optional[Union[str, bool]] = None


@dataclass
class SFTConfig:
    training: TrainingConfig
    dataset: DatasetConfig
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SFTConfig":
        if "training" not in raw:
            raise ConfigError("Missing `training` configuration block.")
        if "dataset" not in raw:
            raise ConfigError("Missing `dataset` configuration block.")

        # Merge optimizer fields into training if present
        training_data = raw["training"].copy()
        if "optimizer" in raw:
            opt = raw["optimizer"]
            training_data.setdefault("learning_rate", opt.get("learning_rate", 2e-4))
            training_data.setdefault("weight_decay", opt.get("weight_decay", 0.0))
            if "betas" in opt and opt["betas"]:
                training_data.setdefault("adam_beta1", opt["betas"][0])
                training_data.setdefault("adam_beta2", opt["betas"][1])
            if "epsilon" in opt:
                training_data.setdefault("adam_epsilon", opt["epsilon"])

        training = TrainingConfig(**training_data)
        dataset = DatasetConfig(**raw["dataset"])
        lora = LoRAConfig(**raw.get("lora", {}))

        config = cls(training=training, dataset=dataset, lora=lora)
        dataset.validate()
        return config


def _load_raw_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    if config_path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ConfigError("PyYAML is not installed. Install it with `pip install pyyaml`.")
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    elif config_path.suffix.lower() == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    else:
        raise ConfigError("Unsupported config format. Use .yaml/.yml or .json files.")

    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be a mapping/object.")
    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT training driver.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML/JSON configuration file describing the training run.",
    )
    return parser.parse_args()


def _load_training_dataset(cfg: DatasetConfig) -> Dataset:
    dataset = load_text_dataset(cfg.name_or_path, split=cfg.split)
    print(f"Loaded dataset from `{cfg.name_or_path}` split `{cfg.split}` with {len(dataset)} examples.")
    return dataset


def _format_example(example: Dict[str, Any]) -> Dict[str, str]:
    prompt = example["prompt"].rstrip()
    response = example["response"].strip()
    return {"text": f"{prompt}\n\nAssistant: {response}"}


def _build_lora_config(cfg: LoRAConfig) -> LoraConfig:
    target_modules = cfg.target_modules if cfg.target_modules is not None else "all-linear"
    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        target_modules=target_modules,
        lora_dropout=cfg.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def _load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(training_cfg: TrainingConfig) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(training_cfg.model_name_or_path)
    if training_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model


def _resolve_resume_checkpoint(
    resume_option: Optional[Union[str, bool]], output_dir: Union[str, Path]
) -> Optional[str]:
    if resume_option in (None, False):
        return None

    output_dir = Path(output_dir)
    if isinstance(resume_option, bool):
        resume_option = "latest"

    if isinstance(resume_option, str):
        if resume_option.lower() == "latest":
            checkpoint = get_last_checkpoint(output_dir)
            if checkpoint is None:
                print(f"No checkpoint found in `{output_dir}`; starting fresh.")
            else:
                print(f"Resuming from latest checkpoint at `{checkpoint}`.")
            return checkpoint

        checkpoint_path = Path(resume_option)
        if not checkpoint_path.exists():
            candidate = output_dir / resume_option
            if candidate.exists():
                checkpoint_path = candidate
            else:
                raise ConfigError(f"Checkpoint not found at `{resume_option}`.")
        resolved = str(checkpoint_path.resolve())
        print(f"Resuming from checkpoint `{resolved}`.")
        return resolved

    raise ConfigError("`resume_from_checkpoint` must be a bool, 'latest', or a valid path.")


def run_training(config: SFTConfig, dataset: Dataset) -> None:
    set_seed(config.training.seed)
    Path(config.training.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(config.training.model_name_or_path)
    model = _load_model(config.training)

    # Sync model config with tokenizer special tokens
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    lora_config = _build_lora_config(config.lora)

    processed_dataset = dataset.map(
        _format_example,
        remove_columns=dataset.column_names,
        desc="Formatting prompts for SFT",
    )

    save_strategy = "steps" if config.training.save_steps else "epoch"
    tc = config.training

    training_args_kwargs = {
        "output_dir": tc.output_dir,
        "per_device_train_batch_size": tc.per_device_train_batch_size,
        "gradient_accumulation_steps": tc.gradient_accumulation_steps,
        "warmup_ratio": tc.warmup_ratio,
        "learning_rate": tc.learning_rate,
        "weight_decay": tc.weight_decay,
        "logging_steps": tc.logging_steps,
        "save_strategy": save_strategy,
        "gradient_checkpointing": tc.gradient_checkpointing,
        "remove_unused_columns": False,
    }

    if tc.log_with_tensorboard:
        logging_dir = tc.logging_dir or str(Path(tc.output_dir) / "tensorboard")
        Path(logging_dir).mkdir(parents=True, exist_ok=True)
        training_args_kwargs["logging_dir"] = logging_dir
        training_args_kwargs["report_to"] = ["tensorboard"]
        print(f"TensorBoard logging enabled. Directory: {logging_dir}")
    else:
        training_args_kwargs["report_to"] = "none"

    # Add optional parameters
    if tc.num_train_epochs is not None:
        training_args_kwargs["num_train_epochs"] = tc.num_train_epochs
    if tc.max_steps is not None:
        training_args_kwargs["max_steps"] = tc.max_steps
    if tc.save_steps is not None:
        training_args_kwargs["save_steps"] = tc.save_steps
    if tc.adam_beta1 is not None:
        training_args_kwargs["adam_beta1"] = tc.adam_beta1
    if tc.adam_beta2 is not None:
        training_args_kwargs["adam_beta2"] = tc.adam_beta2
    if tc.adam_epsilon is not None:
        training_args_kwargs["adam_epsilon"] = tc.adam_epsilon

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    resume_path = _resolve_resume_checkpoint(tc.resume_from_checkpoint, tc.output_dir)
    trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model()
    tokenizer.save_pretrained(config.training.output_dir)


def main() -> None:
    args = parse_args()
    raw_config = _load_raw_config(args.config)
    config = SFTConfig.from_dict(raw_config)

    print("Loaded SFT config:")
    print(json.dumps(raw_config, indent=2))

    dataset = _load_training_dataset(config.dataset)
    sample = dataset[0]
    print(f"\nSample example:\nprompt: {sample['prompt']}\nresponse: {sample['response']}")
    print("\nStarting supervised fine-tuning run...\n")

    run_training(config, dataset)


if __name__ == "__main__":
    main()
