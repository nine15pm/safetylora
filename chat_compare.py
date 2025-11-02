#!/usr/bin/env python3
"""Terminal chat helper to compare base model vs. SFT LoRA adapter."""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def prompt_value(label, current, caster=None):
    suffix = f" [{current}]" if current not in (None, "") else ""
    raw = input(f"{label}{suffix}: ").strip()
    if not raw:
        return current
    if caster is None:
        return raw
    try:
        return caster(raw)
    except ValueError:
        print(f"Invalid {label.lower()}, keeping {current}.")
        return current


def load_models(base: str, adapter: str):
    tok = AutoTokenizer.from_pretrained(adapter or base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    common = dict(dtype="auto", device_map="auto")
    base_model = AutoModelForCausalLM.from_pretrained(base, **common)
    model = PeftModel.from_pretrained(base_model, adapter)
    adapter_name = next(iter(model.peft_config))
    model.set_adapter(adapter_name)
    model.eval()
    return tok, model, adapter_name


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
    parser.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", default="runs/sft/qwen3-4b/run1")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--system", default=None)
    args = parser.parse_args()

    args.base = prompt_value("Base model", args.base)
    args.adapter = prompt_value("LoRA adapter", args.adapter)
    if not args.adapter:
        raise ValueError("LoRA adapter path is required for comparison.")
    args.max_new_tokens = prompt_value("Max new tokens", args.max_new_tokens, int)
    args.temperature = prompt_value("Temperature", args.temperature, float)
    system_prompt = args.system if args.system is not None else ""
    args.system = prompt_value("System prompt", system_prompt) or None

    tok, model, adapter_name = load_models(args.base, args.adapter)

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
        model_inputs = to_device(prompt_inputs, model)

        with model.disable_adapter():
            base_text = generate(model, tok, model_inputs, args.max_new_tokens, args.temperature)
        sft_text = generate(model, tok, model_inputs, args.max_new_tokens, args.temperature)
        print("\n[Base]\n" + base_text + "\n")
        print("[SFT]\n" + sft_text + "\n")


if __name__ == "__main__":
    main()
