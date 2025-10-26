#!/usr/bin/env python3
"""Terminal chat helper to compare base Qwen 0.6B and an SFT LoRA adapter."""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_models(base: str, adapter: str):
    tok = AutoTokenizer.from_pretrained(adapter or base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    common = dict(dtype="auto", device_map="auto")
    base_model = AutoModelForCausalLM.from_pretrained(base, **common).eval()
    sft_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base, **common),
        adapter,
    ).eval()
    return tok, base_model, sft_model


def to_device(batch, model):
    device = next(model.parameters()).device
    return {k: v.to(device) for k, v in batch.items()}


def generate(model, tok, inputs, max_new_tokens: int, temperature: float):
    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tok.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = max(temperature, 1e-4)
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    return tok.decode(out[0, prompt_len:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Compare base vs. SFT LoRA responses.")
    parser.add_argument("--base", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", default="runs/sft/qwen3-0.6b/local-smoke")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--system", default=None)
    args = parser.parse_args()

    tok, base_model, sft_model = load_models(args.base, args.adapter)

    system = args.system.strip() if args.system else None
    print("Chat ready. Ctrl+C to exit.")
    while True:
        try:
            user = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user:
            continue
        messages = []  # Single-turn: we rebuild from system + latest user only.
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        prompt = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if isinstance(prompt, torch.Tensor):
            prompt_inputs = {
                "input_ids": prompt,
                "attention_mask": prompt.ne(tok.pad_token_id).long(),
            }
        else:
            prompt_inputs = {k: v for k, v in prompt.items()}
            if "attention_mask" not in prompt_inputs:
                prompt_inputs["attention_mask"] = prompt_inputs["input_ids"].ne(tok.pad_token_id).long()
        base_inputs = to_device(prompt_inputs, base_model)
        sft_inputs = to_device(prompt_inputs, sft_model)
        base_text = generate(base_model, tok, base_inputs, args.max_new_tokens, args.temperature)
        sft_text = generate(sft_model, tok, sft_inputs, args.max_new_tokens, args.temperature)
        print("\n[Base]\n" + base_text + "\n")
        print("[SFT]\n" + sft_text + "\n")


if __name__ == "__main__":
    main()
