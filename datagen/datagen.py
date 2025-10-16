from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from datagen.llm import LLM
from datagen.prompts import PROMPTGEN_SYSTEM_PROMPT, PROMPTGEN_USER_PROMPT

PROVIDERS = ("moonshot", "deepseek", "grok")
OUTPUT_DIR = Path("temp")
INTENT_SEEDS_PATH = Path(__file__).with_name("intentseeds.jsonl")


def _build_user_prompt(seed: Dict[str, str], num_messages: int = 1) -> str:
    intent = seed["intent"]
    intent_type = seed["type"]
    category = seed.get("category", "unknown")
    intent_context = f"CATEGORY: {category}\nINTENT: {intent}\nTYPE: {intent_type}"
    return PROMPTGEN_USER_PROMPT.format(
        num_messages=num_messages, intent_context=intent_context
    ).strip()


def _generate_for_provider(provider: str, system_prompt: str, user_prompt: str) -> str:
    llm = LLM(provider=provider, temperature=0.7, max_tokens=512)
    return llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)


def generate_sample(seed: Dict[str, str], num_messages: int = 1) -> Dict[str, str]:
    system_prompt = PROMPTGEN_SYSTEM_PROMPT.strip()
    user_prompt = _build_user_prompt(seed, num_messages)

    results: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(PROVIDERS)) as executor:
        futures = {
            executor.submit(_generate_for_provider, provider, system_prompt, user_prompt): provider
            for provider in PROVIDERS
        }
        for future in as_completed(futures):
            provider = futures[future]
            try:
                results[provider] = future.result()
            except Exception as exc:  # capture auth errors etc. for quick debugging
                results[provider] = f"ERROR: {exc}"
    return results


def _load_intent_seeds(path: Path = INTENT_SEEDS_PATH) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_results(seed: Dict[str, str], outputs: Dict[str, str]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    output_path = OUTPUT_DIR / f"llm_sample_{timestamp}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for provider, content in outputs.items():
            record = {
                "provider": provider,
                "category": seed.get("category"),
                "intent": seed["intent"],
                "type": seed["type"],
                "content": content,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def main() -> None:
    seeds = _load_intent_seeds()
    seed = random.choice(seeds)
    outputs = generate_sample(seed=seed, num_messages=1)
    path = _write_results(seed, outputs)
    print(f"Wrote comparison sample to {path} for seed intent: {seed['intent']}")


if __name__ == "__main__":
    main()
