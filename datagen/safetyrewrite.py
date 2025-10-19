"""Async safety rewriting for unsafe assistant responses using Gemini."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from datagen.llm import GeminiCompletion
from datagen.prompts import SAFETY_REWRITE_SYSTEM_PROMPT, SAFETY_REWRITE_USER_PROMPT


# Core async pattern
async def run_concurrent_iter(tasks: list, max_concurrent: int):
    """Run tasks with concurrency limit, yielding results as they complete."""
    sem = asyncio.Semaphore(max_concurrent)

    async def _run_with_sem(task):
        async with sem:
            return await task

    for coro in asyncio.as_completed([_run_with_sem(task) for task in tasks]):
        yield await coro


# I/O utilities
def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL file into list of dicts."""
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Append records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# Config
@dataclass
class RewriteConfig:
    temperature: float
    max_tokens: int
    max_concurrent: int = 4
    resume: bool = True
    safety_policy_file: str = "safetypolicy.jsonl"
    system_prompts_file: str = "sysprompts.jsonl"
    unsafe_turns_file: str = "assistantturns_unsafe.jsonl"
    rewritten_file: str = "assistantturns_rewritten.jsonl"

    @property
    def safety_policy_path(self) -> Path:
        return Path(__file__).parent / self.safety_policy_file

    @property
    def system_prompts_path(self) -> Path:
        return Path(__file__).parent / self.system_prompts_file

    @property
    def unsafe_turns_path(self) -> Path:
        return Path(__file__).parent / self.unsafe_turns_file

    @property
    def rewritten_path(self) -> Path:
        return Path(__file__).parent / self.rewritten_file


# Rewrite all unsafe turns
async def rewrite_unsafe(config: RewriteConfig) -> Path:
    """Rewrite all unsafe assistant turns to be safe."""
    # Load data
    unsafe_turns = read_jsonl(config.unsafe_turns_path)
    policies_list = read_jsonl(config.safety_policy_path)
    policies = {p["category"]: p for p in policies_list if "category" in p}
    system_prompts_list = read_jsonl(config.system_prompts_path)
    system_prompts = {sp["id"]: sp["prompt"] for sp in system_prompts_list if "id" in sp}

    # Build jobs - only process entries marked as unsafe
    jobs = []
    for turn in unsafe_turns:
        if not turn.get("user_msg") or not turn.get("assistant_msg"):
            continue
        if turn.get("judge_safe") != 0:
            continue

        jobs.append({
            "turn": turn,
            "policy": policies.get(turn.get("category", ""), {}),
        })

    # Filter completed if resuming
    if config.resume:
        completed_ids = {
            row["assistant_turn_id"]
            for row in read_jsonl(config.rewritten_path)
            if row.get("assistant_turn_id") and "rewritten_msg" in row
        }
        jobs = [j for j in jobs if j["turn"]["assistant_turn_id"] not in completed_ids]

    if not jobs:
        print("No jobs to process")
        return config.rewritten_path

    # Create model
    model = None
    try:
        model = GeminiCompletion(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )

        # Rewrite one turn
        async def _rewrite_one(job):
            turn = job["turn"]

            # Get system prompt text from ID
            sys_prompt_id = turn.get("system_prompt_id", "")
            sys_prompt_text = system_prompts.get(sys_prompt_id, "You are a helpful assistant.")

            # Assemble context as JSON objects
            policy_json = json.dumps(
                {"category": turn.get("category", "unknown"), **job["policy"]},
                ensure_ascii=False,
            )
            system_prompt_json = json.dumps(
                {"system_prompt": sys_prompt_text},
                ensure_ascii=False,
            )
            user_prompt_json = json.dumps(
                {"user_prompt": turn['user_msg']},
                ensure_ascii=False,
            )
            assistant_response_json = json.dumps(
                {"assistant_response": turn['assistant_msg']},
                ensure_ascii=False,
            )
            unsafe_reason_json = json.dumps(
                {"unsafe_reason": turn.get('judge_reason', 'none')},
                ensure_ascii=False,
            )

            # Assemble context for rewrite prompt
            user_prompt = SAFETY_REWRITE_USER_PROMPT.format(
                safety_policy=policy_json,
                system_prompt=system_prompt_json,
                user_prompt=user_prompt_json,
                assistant_response=assistant_response_json,
                unsafe_reason=unsafe_reason_json,
            )

            try:
                response = await model.generate(
                    system_prompt=SAFETY_REWRITE_SYSTEM_PROMPT.strip(),
                    user_prompt=user_prompt,
                )
                return {**turn, "rewritten_msg": response, "rewrite_provider": "gemini"}
            except Exception as e:
                return {**turn, "rewritten_msg": None, "rewrite_error": str(e), "rewrite_provider": "gemini"}

        # Run concurrent rewriting
        tasks = [_rewrite_one(job) for job in jobs]
        total = 0

        with tqdm(total=len(jobs), desc="Rewriting unsafe responses", unit="turn") as pbar:
            async for result in run_concurrent_iter(tasks, config.max_concurrent):
                write_jsonl(config.rewritten_path, [result])
                total += 1
                pbar.update(1)

        print(f"Wrote {total} rewritten records to {config.rewritten_path}")
        return config.rewritten_path
    finally:
        if model is not None:
            await model.aclose()


__all__ = ["RewriteConfig", "rewrite_unsafe"]
