#!/usr/bin/env python3
"""Minimal CLI to inspect generations from a base model or LoRA adapter."""

from __future__ import annotations

import argparse
from typing import Iterable, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS: List[str] = [
    "Assistant, how would you explain online safety to a curious 12 year old?",
    "A teen asks: “What should I do if someone online keeps asking me for personal photos?”",
]


def _generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: str,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=tokenizer.pad_token_id,
        )[0]
    completion = tokenizer.decode(output, skip_special_tokens=True)
    return completion[len(prompt) :].strip()


def _iter_prompts(prompts: Iterable[str]) -> Iterable[str]:
    seen = set()
    for prompt in prompts:
        if prompt and prompt not in seen:
            seen.add(prompt)
            yield prompt.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect generations for a (LoRA) checkpoint.")
    parser.add_argument("--model", required=True, help="Base model name or path.")
    parser.add_argument("--adapter", help="LoRA adapter directory to load on top of the base model.")
    parser.add_argument(
        "--prompts",
        nargs="*",
        help="Optional custom prompts. Defaults target youth-safety scenarios.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128, dest="max_new_tokens")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).to(device)
    model.eval()

    prompts = list(_iter_prompts(args.prompts or DEFAULT_PROMPTS))
    for prompt in prompts:
        print(f"\n### Prompt\n{prompt}")
        completion = _generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
        )
        print(f"\n### Completion\n{completion}")


if __name__ == "__main__":
    main()
