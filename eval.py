"""Tiny evaluation harness for LoRA checkpoints."""

import json
import sys
import time
from datetime import datetime
from importlib import import_module
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(base_model_id: str, adapter_path: str | None):
    """Load model with optional LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, dtype="auto", device_map="auto"
    ).eval()

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path).eval()

    return model, tokenizer


def format_prompt(item: dict, tokenizer) -> str:
    """Format prompt into chat template."""
    if "messages" in item:
        messages = item["messages"]
    else:
        messages = []
        if "system" in item:
            messages.append({"role": "system", "content": item["system"]})
        messages.append({"role": "user", "content": item["prompt"]})

    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
    )


def run_eval(
    base_model: str,
    prompts_path: str,
    eval_fn: callable,
    adapter_path: str | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    run_id: str | None = None,
    output_dir: str = "evals/results",
):
    """Run evaluation and save results."""
    # Load model and prompts
    model, tokenizer = load_model_and_tokenizer(base_model, adapter_path)
    with open(prompts_path, encoding="utf-8") as f:
        if prompts_path.endswith(".json"):
            data = json.load(f)
            prompts = data if isinstance(data, list) else data.get("prompts", [])
        else:
            prompts = [json.loads(line) for line in f if line.strip()]

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    # Generate and evaluate
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        formatted = [format_prompt(item, tokenizer) for item in batch]

        inputs = tokenizer(formatted, padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"].to(model.device),
                pad_token_id=tokenizer.pad_token_id,
                **gen_kwargs
            )
        elapsed = time.time() - start

        for j, (item, output) in enumerate(zip(batch, outputs)):
            response = tokenizer.decode(output[input_ids.shape[-1]:], skip_special_tokens=True).strip()
            eval_result = eval_fn(item, response)
            results.append({
                "prompt": item.get("prompt", ""),
                "response": response,
                "eval": eval_result,
                "latency_ms": round(elapsed / len(batch) * 1000, 1),
            })

    # Aggregate metrics: any numeric field from eval_fn results gets averaged
    # e.g. if eval_fn returns {"score": 1} for correct and {"score": 0} for incorrect,
    # the final metric will be the accuracy (avg of 1s and 0s)
    metrics = {}
    for r in results:
        for k, v in r["eval"].items():
            if isinstance(v, (int, float)):
                metrics.setdefault(k, []).append(float(v))
    metrics = {k: round(sum(v) / len(v), 3) for k, v in metrics.items()}

    # Save results
    timestamp = datetime.utcnow().isoformat().replace(":", "-") + "Z"
    run_id = run_id or (Path(adapter_path).parent.name if adapter_path else "base")
    detail_path = Path(output_dir) / run_id / f"{timestamp}.jsonl"
    detail_path.parent.mkdir(parents=True, exist_ok=True)

    with open(detail_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "run_id": run_id,
        "timestamp": timestamp,
        "base_model": base_model,
        "adapter": adapter_path,
        "prompts": prompts_path,
        "num_samples": len(results),
        "metrics": metrics,
        "details": str(detail_path),
    }
    log_path = Path(output_dir) / "eval_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Evaluated {len(results)} prompts | Metrics: {metrics}")
    print(f"Details: {detail_path}")
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eval.py <config.yaml>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)

    # Load eval function
    module_path, fn_name = config["eval_fn"].split(":")
    eval_fn = getattr(import_module(module_path), fn_name)

    run_eval(
        base_model=config["base_model"],
        adapter_path=config.get("adapter"),
        prompts_path=config["prompts"],
        eval_fn=eval_fn,
        batch_size=config.get("batch_size", 4),
        max_new_tokens=config.get("max_new_tokens", 256),
        temperature=config.get("temperature", 0.7),
        run_id=config.get("run_id"),
        output_dir=config.get("output_dir", "evals/results"),
    )
