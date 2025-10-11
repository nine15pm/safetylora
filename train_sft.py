#!/usr/bin/env python3
"""Configuration loader for the SFT LoRA pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
except ImportError:  # pragma: no cover - import guard for environments without PyYAML
    yaml = None


class ConfigError(ValueError):
    """Raised when the training configuration is invalid."""


@dataclass
class DatasetConfig:
    name_or_path: Optional[str] = None
    split: str = "train"
    synthetic_n_examples: Optional[int] = None

    def validate(self) -> None:
        has_source = self.name_or_path is not None
        wants_synthetic = self.synthetic_n_examples is not None

        if has_source and wants_synthetic:
            raise ConfigError(
                "Provide either `dataset.name_or_path` or `dataset.synthetic_n_examples`, not both."
            )

        if not has_source and not wants_synthetic:
            raise ConfigError(
                "Specify a dataset via `dataset.name_or_path` or request a synthetic dataset "
                "with `dataset.synthetic_n_examples`."
            )

        if wants_synthetic and self.synthetic_n_examples is not None:
            if self.synthetic_n_examples <= 0:
                raise ConfigError("`dataset.synthetic_n_examples` must be a positive integer.")


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[Union[str, List[str]]] = None

    def validate(self) -> None:
        if self.r <= 0:
            raise ConfigError("`lora.r` must be a positive integer.")
        if self.alpha <= 0:
            raise ConfigError("`lora.alpha` must be a positive integer.")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("`lora.dropout` must be between 0 and 1 (exclusive of 1).")
        if self.target_modules is not None:
            if isinstance(self.target_modules, str):
                if not self.target_modules.strip():
                    raise ConfigError("`lora.target_modules` string cannot be empty.")
            else:
                if not self.target_modules:
                    raise ConfigError("`lora.target_modules` list cannot be empty.")
                if not all(isinstance(module, str) and module.strip() for module in self.target_modules):
                    raise ConfigError("`lora.target_modules` entries must be non-empty strings.")


@dataclass
class OptimizerConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    betas: Optional[List[float]] = None
    epsilon: Optional[float] = None

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ConfigError("`optimizer.learning_rate` must be greater than 0.")
        if self.weight_decay < 0:
            raise ConfigError("`optimizer.weight_decay` cannot be negative.")
        if self.betas is not None:
            if len(self.betas) != 2:
                raise ConfigError("`optimizer.betas` must contain exactly two floats (beta1, beta2).")
            if not all(0.0 < beta < 1.0 for beta in self.betas):
                raise ConfigError("`optimizer.betas` values must be in the open interval (0, 1).")
        if self.epsilon is not None and self.epsilon <= 0:
            raise ConfigError("`optimizer.epsilon` must be greater than 0.")


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

    def validate(self) -> None:
        if not self.output_dir:
            raise ConfigError("`training.output_dir` is required.")
        if not self.model_name_or_path:
            raise ConfigError("`training.model_name_or_path` is required.")
        if self.per_device_train_batch_size <= 0:
            raise ConfigError("`training.per_device_train_batch_size` must be positive.")
        if self.gradient_accumulation_steps <= 0:
            raise ConfigError("`training.gradient_accumulation_steps` must be positive.")

        if self.num_train_epochs is None and self.max_steps is None:
            raise ConfigError("Specify `training.num_train_epochs` or `training.max_steps`.")
        if self.num_train_epochs is not None and self.num_train_epochs <= 0:
            raise ConfigError("`training.num_train_epochs` must be positive.")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ConfigError("`training.max_steps` must be positive.")

        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ConfigError("`training.warmup_ratio` must be between 0 and 1 (exclusive of 1).")


@dataclass
class SFTConfig:
    training: TrainingConfig
    dataset: DatasetConfig
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SFTConfig":
        try:
            training = TrainingConfig(**raw["training"])
        except KeyError as exc:
            raise ConfigError("Missing `training` configuration block.") from exc

        try:
            dataset = DatasetConfig(**raw["dataset"])
        except KeyError as exc:
            raise ConfigError("Missing `dataset` configuration block.") from exc

        lora_cfg = LoRAConfig(**raw.get("lora", {}))
        optimizer_cfg = OptimizerConfig(**raw.get("optimizer", {}))

        config = cls(
            training=training,
            dataset=dataset,
            lora=lora_cfg,
            optimizer=optimizer_cfg,
        )
        config.validate()
        return config

    def validate(self) -> None:
        self.training.validate()
        self.dataset.validate()
        self.lora.validate()
        self.optimizer.validate()


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


def main() -> None:
    args = parse_args()
    raw_config = _load_raw_config(args.config)
    config = SFTConfig.from_dict(raw_config)

    print("Loaded SFT config:")
    print(json.dumps(raw_config, indent=2))
    print(
        "\nTraining pipeline is not yet implemented. "
        "Next step: integrate TRL SFTTrainer using this configuration."
    )


if __name__ == "__main__":
    main()
