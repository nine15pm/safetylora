"""Minimal dataset loader enforcing the prompt/response contract for SFT/GRPO."""

from __future__ import annotations

from typing import Iterable, Sequence

from datasets import Dataset, DatasetDict, load_dataset

REQUIRED_COLUMNS: Sequence[str] = ("prompt", "response")


def _ensure_columns(dataset: Dataset, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """Raise if the dataset is missing required text columns."""
    missing = [column for column in required if column not in dataset.column_names]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def load_text_dataset(source: str, *, split: str = "train", **load_kwargs) -> Dataset:
    """
    Load a dataset expected to expose `prompt` and `response` string fields.

    `source` can be a local file (e.g. `json`/`parquet`) or a Hugging Face hub name.
    Additional `load_dataset` keyword arguments (like `data_files`) pass through.
    """
    loaded = load_dataset(path=source, split=split, **load_kwargs)
    if isinstance(loaded, DatasetDict):  # pragma: no cover - defensive; split usually returns Dataset
        dataset = loaded[split]
    else:
        dataset = loaded

    _ensure_columns(dataset)
    return dataset


__all__ = ["load_text_dataset", "REQUIRED_COLUMNS"]
