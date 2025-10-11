"""Synthetic prompt/response dataset for quick smoke tests.

This module is a temporary helper while we wire the training stack; replace with
real datasets once available.
"""

from __future__ import annotations

from typing import List, Tuple

from datasets import Dataset


_BASE_pairs: Tuple[Tuple[str, str], ...] = (
    (
        "User: Give me a playful explanation of {topic} that is safe for kids.",
        "Assistant: Sure! {topic} is like a friendly adventure where we learn something new in a fun, gentle way.",
    ),
    (
        "User: Tell a short, positive story about {topic} that is age-appropriate.",
        "Assistant: Absolutely! Imagine {topic} as a bright story where everyone stays kind, curious, and safe.",
    ),
    (
        "User: How do I teach {topic} while making sure it stays wholesome?",
        "Assistant: Start with simple, caring language and focus on the helpful lessons hidden inside {topic}.",
    ),
    (
        "User: What's the safest way to talk to a young person about {topic}?",
        "Assistant: Keep it gentle, honest, and encouraging so the conversation about {topic} feels supportive.",
    ),
)

_TOPICS: Tuple[str, ...] = (
    "online kindness",
    "sharing toys",
    "asking for help",
    "healthy habits",
    "using imagination",
    "teamwork at school",
)


def build_sft_synthetic_dataset(num_examples: int) -> Dataset:
    """
    Return a small Dataset with `prompt` and `response` columns for smoke testing.

    The content is intentionally simple and youth-safe, mirroring the structure expected
    by the real training data while remaining fast to generate.
    """
    if num_examples <= 0:
        raise ValueError("`num_examples` must be positive.")

    prompts: List[str] = []
    responses: List[str] = []

    for idx in range(num_examples):
        prompt_template, response_template = _BASE_pairs[idx % len(_BASE_pairs)]
        topic = _TOPICS[idx % len(_TOPICS)]
        prompts.append(prompt_template.format(topic=topic))
        responses.append(response_template.format(topic=topic))

    return Dataset.from_dict({"prompt": prompts, "response": responses})


__all__ = ["build_sft_synthetic_dataset"]
