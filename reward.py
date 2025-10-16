"""Reward scoring interfaces for the GRPO stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import torch


class RewardScorer:
    """Abstract scorer returning per-sample rewards."""

    def score(self, prompts: Sequence[str], completions: Sequence[str]) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class ConstantRewardScorer(RewardScorer):
    """Return a fixed reward for every sample. Useful for smoke tests."""

    value: float

    def score(self, prompts: Sequence[str], completions: Sequence[str]) -> torch.Tensor:
        if len(prompts) != len(completions):
            raise ValueError("Prompts and completions must share the same batch dimension.")
        batch_size = len(prompts)
        return torch.full((batch_size,), self.value, dtype=torch.float32)


def create_reward_scorer(judge_name: str, *, stub_score: Optional[float] = None) -> RewardScorer:
    """
    Factory for reward scorers.

    Currently exposes a constant-score stub that we can replace with a real judge later.
    """

    if judge_name == "stub-judge":
        value = 0.0 if stub_score is None else stub_score
        return ConstantRewardScorer(value=value)

    raise NotImplementedError(
        f"Reward judge `{judge_name}` is not implemented yet. Provide a stub or extend the factory."
    )


def build_reward_function(scorer: RewardScorer) -> Callable[[Sequence[str], Sequence[str], Dict], torch.Tensor]:
    """
    Wrap a `RewardScorer` into the callable shape expected by TRL's GRPO trainer.

    TRL forwards prompts, completions, and an optional dict of extra metadata.
    We ignore the metadata for now but keep the signature for compatibility.
    """

    def reward_fn(
        *,
        prompts: Sequence[str],
        completions: Sequence[str],
        completion_ids: Sequence[torch.Tensor],
        **_: Dict,
    ) -> torch.Tensor:
        return scorer.score(prompts, completions)

    return reward_fn


__all__ = ["RewardScorer", "ConstantRewardScorer", "create_reward_scorer", "build_reward_function"]
