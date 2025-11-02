#!/usr/bin/env python3
"""Minimal TRL SFT training script with LoRA adapters."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run supervised fine-tuning with LoRA.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not raw or "training" not in raw or "dataset" not in raw or "lora" not in raw:
        raise ValueError("Config must include `training`, `dataset`, and `lora` sections.")
    return raw


def format_dataset(path: str, split: str) -> Dataset:
    dataset = load_dataset("json", data_files={split: path}, split=split)

    def convert(example: Dict[str, Any]) -> Dict[str, Any]:
        prompt_messages = []
        system_prompt = example.get("sp")
        if system_prompt:
            prompt_messages.append({"role": "system", "content": system_prompt})
        prompt_messages.append({"role": "user", "content": example["prompt"]})
        completion = [{"role": "assistant", "content": example["response"]}]
        return {"prompt": prompt_messages, "completion": completion}

    formatted = dataset.map(convert, remove_columns=dataset.column_names)
    return formatted


def build_lora_config(raw_lora: Dict[str, Any]) -> LoraConfig:
    lora_kwargs = dict(raw_lora)
    alpha = lora_kwargs.pop("alpha", None)
    if alpha is not None:
        lora_kwargs["lora_alpha"] = alpha
    lora_kwargs.setdefault("bias", "none")
    return LoraConfig(task_type=TaskType.CAUSAL_LM, **lora_kwargs)


def prepare_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    training_cfg = dict(config["training"])
    dataset_cfg = dict(config["dataset"])
    optimizer_cfg = dict(config.get("optimizer", {}))
    wandb_cfg = dict(config.get("wandb", {}))

    if "output_dir" not in training_cfg:
        raise ValueError("`training.output_dir` must be set in the config.")

    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = training_cfg.get("seed")
    if seed is not None:
        set_seed(seed)

    model_name = training_cfg.get("model_name_or_path")
    if not model_name:
        raise ValueError("`training.model_name_or_path` must be set in the config.")

    tokenizer = prepare_tokenizer(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
    )

    gradient_checkpointing_kwargs = training_cfg.get("gradient_checkpointing_kwargs")
    if training_cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )
        if getattr(model.config, "use_cache", None) is not None:
            model.config.use_cache = False

    split = dataset_cfg.get("split", "train")
    dataset = format_dataset(dataset_cfg["name_or_path"], split)

    print("Sample formatted example:")
    print(json.dumps(dataset[0], indent=2))

    lora_config = build_lora_config(config["lora"])

    training_kwargs = {**training_cfg, **optimizer_cfg}
    special_keys = {
        "model_name_or_path",
        "seed",
        "gradient_checkpointing",
        "gradient_checkpointing_kwargs",
        "resume_from_checkpoint",
    }
    training_kwargs = {k: v for k, v in training_kwargs.items() if k not in special_keys}
    if wandb_cfg:
        env_var_map = {
            "project": "WANDB_PROJECT",
            "entity": "WANDB_ENTITY",
            "group": "WANDB_RUN_GROUP",
            "run_name": "WANDB_NAME",
        }
        for key, env_key in env_var_map.items():
            value = wandb_cfg.get(key)
            if value:
                os.environ.setdefault(env_key, str(value))
        tags = wandb_cfg.get("tags")
        if tags:
            os.environ.setdefault("WANDB_TAGS", ",".join(tags))
        watch = wandb_cfg.get("watch")
        if watch:
            os.environ.setdefault("WANDB_WATCH", str(watch))
        training_kwargs.setdefault("report_to", ["wandb"])
    training_args = TrainingArguments(**training_kwargs)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=training_cfg.get("resume_from_checkpoint"))
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
