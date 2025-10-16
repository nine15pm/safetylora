#!/usr/bin/env python3
"""Config + CLI scaffold for the GRPO stage building on SFT adapters."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None

from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from data_loader import load_text_dataset
from synthetic_dataset import build_sft_synthetic_dataset
from reward import build_reward_function, create_reward_scorer

try:
    from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer
except ImportError as exc:  # pragma: no cover - optional dependency
    TRLGRPOConfig = None
    GRPOTrainer = None
    _TRL_IMPORT_ERROR = exc
else:
    _TRL_IMPORT_ERROR = None


class ConfigError(ValueError):
    """Raised when the GRPO configuration is invalid."""


@dataclass
class DatasetConfig:
    name_or_path: Optional[str] = None
    split: str = "train"
    synthetic_n_prompts: Optional[int] = None

    def validate(self) -> None:
        has_source = self.name_or_path is not None
        wants_synthetic = self.synthetic_n_prompts is not None

        if has_source and wants_synthetic:
            raise ConfigError(
                "Provide either `dataset.name_or_path` or `dataset.synthetic_n_prompts`, not both."
            )
        if not has_source and not wants_synthetic:
            raise ConfigError(
                "Specify a dataset via `dataset.name_or_path` or `dataset.synthetic_n_prompts`."
            )


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[Union[str, List[str]]] = None


@dataclass
class RewardConfig:
    """
    Configuration for LLM-judge rewards.

    `judge_model_name` points to the language model that will evaluate rollouts.
    `stub_score` allows local smoke tests without invoking an actual judge.
    """

    judge_model_name: str = "stub-judge"
    prompt_template: Optional[str] = None
    completion_template: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    batch_size: int = 1
    stub_score: Optional[float] = None

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ConfigError("`reward.batch_size` must be positive.")
        if not (0.0 <= self.top_p <= 1.0):
            raise ConfigError("`reward.top_p` must fall within [0, 1].")
        if self.temperature < 0.0:
            raise ConfigError("`reward.temperature` must be non-negative.")

        if self.stub_score is not None and not (-10.0 <= self.stub_score <= 10.0):
            raise ConfigError("`reward.stub_score` should be in [-10, 10] for stability.")


@dataclass
class TrainingConfig:
    output_dir: str
    policy_model_name_or_path: str
    sft_adapter_path: str
    seed: int = 42
    gradient_checkpointing: bool = True
    resume_from_checkpoint: Optional[Union[str, bool]] = None
    trl_grpo: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.output_dir:
            raise ConfigError("`training.output_dir` must be provided.")
        if not self.policy_model_name_or_path:
            raise ConfigError("`training.policy_model_name_or_path` must be provided.")
        if not self.sft_adapter_path:
            raise ConfigError("`training.sft_adapter_path` must be provided.")


@dataclass
class GRPOConfig:
    training: TrainingConfig
    dataset: DatasetConfig
    reward: RewardConfig = field(default_factory=RewardConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "GRPOConfig":
        if "training" not in raw:
            raise ConfigError("Missing `training` configuration block.")
        if "dataset" not in raw:
            raise ConfigError("Missing `dataset` configuration block.")

        training = TrainingConfig(**raw["training"])
        dataset = DatasetConfig(**raw["dataset"])
        reward_cfg = RewardConfig(**raw.get("reward", {}))
        lora_cfg = LoRAConfig(**raw.get("lora", {}))

        training.validate()
        dataset.validate()
        reward_cfg.validate()

        return cls(training=training, dataset=dataset, reward=reward_cfg, lora=lora_cfg)


def _prepare_prompt_dataset(cfg: DatasetConfig) -> Dataset:
    """Return a `prompt`-only dataset for GRPO rollouts."""

    if cfg.synthetic_n_prompts is not None:
        dataset = build_sft_synthetic_dataset(cfg.synthetic_n_prompts)
        # NOTE: synthetic toggle is temporary and will be removed once real data lands.
    else:
        if cfg.name_or_path is None:  # pragma: no cover - defensive guard, validate() should catch
            raise ConfigError("`dataset.name_or_path` must be provided when not using synthetic prompts.")
        dataset = load_text_dataset(cfg.name_or_path, split=cfg.split)

    if "prompt" not in dataset.column_names:
        raise ConfigError("Prompt dataset must include a `prompt` column.")

    prompt_only = dataset.select_columns(["prompt"])
    if len(prompt_only) == 0:
        raise ConfigError("Prompt dataset is empty; provide at least one prompt.")

    return prompt_only


def _load_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_policy_model(cfg: TrainingConfig) -> PeftModel:
    if not Path(cfg.sft_adapter_path).exists():
        raise ConfigError(
            f"SFT adapter not found at `{cfg.sft_adapter_path}`. Provide a valid adapter checkpoint."
        )

    model = AutoModelForCausalLM.from_pretrained(cfg.policy_model_name_or_path)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    peft_model = PeftModel.from_pretrained(
        model,
        cfg.sft_adapter_path,
        is_trainable=True,
    )
    return peft_model


def _prepare_trl_args(training_cfg: TrainingConfig) -> Dict[str, Any]:
    """Merge core training outputs with GRPO-specific arguments."""

    trl_args = dict(training_cfg.trl_grpo)
    trl_args.setdefault("output_dir", training_cfg.output_dir)
    return trl_args


def _resolve_resume_checkpoint(
    resume_option: Optional[Union[str, bool]], output_dir: Union[str, Path]
) -> Optional[str]:
    if resume_option in (None, False):
        return None

    if isinstance(resume_option, str):
        return resume_option

    # resume_option is True – try to find last checkpoint under the output dir
    last_checkpoint = get_last_checkpoint(str(output_dir))
    return last_checkpoint


def _load_raw_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ConfigError("PyYAML is not installed. Install it with `pip install pyyaml`.")
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    elif suffix == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    else:
        raise ConfigError("Unsupported config format. Use .yaml/.yml or .json files.")

    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be a mapping/object.")
    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA GRPO training driver.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML/JSON configuration file describing the GRPO run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_config = _load_raw_config(args.config)
    config = GRPOConfig.from_dict(raw_config)

    print("Loaded GRPO config:")
    print(json.dumps(raw_config, indent=2))

    prompt_dataset = _prepare_prompt_dataset(config.dataset)
    dataset_source = (
        config.dataset.name_or_path
        if config.dataset.name_or_path is not None
        else f"synthetic:{config.dataset.synthetic_n_prompts}"
    )
    print(
        f"Prepared prompt dataset from {dataset_source} with {len(prompt_dataset)} prompts."
    )

    reward_scorer = create_reward_scorer(
        config.reward.judge_model_name, stub_score=config.reward.stub_score
    )
    reward_fn = build_reward_function(reward_scorer)
    stub_note = (
        f"stub_score={config.reward.stub_score}"
        if config.reward.stub_score is not None
        else "default stub_score=0.0"
    )
    print(
        f"Initialized reward scorer `{config.reward.judge_model_name}` ({stub_note})."
    )
    print("Reward callable ready for GRPO trainer.")

    tokenizer = _load_tokenizer(config.training.policy_model_name_or_path)
    print(
        f"Loaded tokenizer from `{config.training.policy_model_name_or_path}` "
        f"with pad token `{tokenizer.pad_token}`."
    )

    policy_model = _load_policy_model(config.training)
    policy_model.config.pad_token_id = tokenizer.pad_token_id
    adapter_names = list(policy_model.peft_config.keys())
    print(
        f"Loaded policy model with adapters {adapter_names} from `{config.training.sft_adapter_path}`."
    )

    trl_args_dict = _prepare_trl_args(config.training)
    print(f"Prepared GRPO config arguments with keys: {sorted(trl_args_dict.keys())}.")

    if _TRL_IMPORT_ERROR is not None or TRLGRPOConfig is None:
        raise ConfigError(
            "`trl` is not installed. Install it (e.g. `pip install trl`) before running GRPO training."
        ) from _TRL_IMPORT_ERROR

    try:
        trl_args = TRLGRPOConfig(**trl_args_dict)
    except Exception as exc:  # pragma: no cover - diagnostic path
        raise ConfigError(f"Failed to build GRPOConfig: {exc}") from exc
    print("GRPOConfig instantiation succeeded; launching GRPOTrainer.")

    set_seed(config.training.seed)
    Path(config.training.output_dir).mkdir(parents=True, exist_ok=True)

    resume_checkpoint = _resolve_resume_checkpoint(
        config.training.resume_from_checkpoint, config.training.output_dir
    )
    if resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_checkpoint}")

    trainer = GRPOTrainer(
        model=policy_model,
        reward_funcs=reward_fn,
        args=trl_args,
        train_dataset=prompt_dataset,
        processing_class=tokenizer,
    )

    train_output = trainer.train(resume_from_checkpoint=resume_checkpoint)
    print(f"Training complete. global_step={trainer.state.global_step}")
    print(f"Train metrics: {train_output.metrics}")

    trainer.accelerator.wait_for_everyone()
    trainer.save_model()
    trainer.save_state()
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(config.training.output_dir)
    print(f"Saved GRPO adapter and tokenizer to `{config.training.output_dir}`.")


if __name__ == "__main__":
    main()
