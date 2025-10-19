"""Async safety judging for assistant responses using Gemini."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from datagen.llm import GeminiCompletion
from datagen.prompts import SAFETY_JUDGE_SYSTEM_PROMPT, SAFETY_JUDGE_USER_PROMPT


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


def parse_judge_response(text: str) -> tuple[int, str]:
    """Parse judge JSON response into (safe_score, reason)."""
    # Strip markdown code blocks if present
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()

    # Parse JSON
    data = json.loads(text)
    safe = int(data["safe"])
    reason = data.get("reason", "").strip() or "none"

    return safe, reason


# Config
@dataclass
class JudgeConfig:
    temperature: float
    max_tokens: int
    max_concurrent: int = 4
    resume: bool = True
    safety_policy_file: str = "safetypolicy.jsonl"
    assistant_turns_file: str = "assistantturns.jsonl"
    scored_file: str = "assistantturns_scored.jsonl"

    @property
    def safety_policy_path(self) -> Path:
        return Path(__file__).parent / self.safety_policy_file

    @property
    def assistant_turns_path(self) -> Path:
        return Path(__file__).parent / self.assistant_turns_file

    @property
    def scored_path(self) -> Path:
        return Path(__file__).parent / self.scored_file


# Judge all turns
async def judge_safety(config: JudgeConfig) -> Path:
    """Judge all assistant turns for safety violations."""
    # Load data
    assistant_turns = read_jsonl(config.assistant_turns_path)
    policies_list = read_jsonl(config.safety_policy_path)
    policies = {p["category"]: p for p in policies_list if "category" in p}

    # Build jobs
    jobs = []
    for turn in assistant_turns:
        if not turn.get("user_msg") or not turn.get("assistant_msg"):
            continue

        jobs.append({
            "turn": turn,
            "policy": policies.get(turn.get("category", ""), {}),
        })

    # Filter completed if resuming
    if config.resume:
        completed_ids = {
            row["assistant_turn_id"]
            for row in read_jsonl(config.scored_path)
            if row.get("assistant_turn_id") and "judge_safe" in row
        }
        jobs = [j for j in jobs if j["turn"]["assistant_turn_id"] not in completed_ids]

    if not jobs:
        print("No jobs to process")
        return config.scored_path

    # Create model
    model = GeminiCompletion(
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
    )

    # Judge one turn
    async def _judge_one(job):
        turn = job["turn"]
        policy_str = json.dumps(
            {"category": turn.get("category", "unknown"), **job["policy"]},
            ensure_ascii=False,
        )
        user_prompt = SAFETY_JUDGE_USER_PROMPT.format(
            safety_policy=policy_str,
            user_prompt=f"USER PROMPT:\n{turn['user_msg']}",
            assistant_response=f"ASSISTANT RESPONSE:\n{turn['assistant_msg']}",
        )

        try:
            response = await model.generate(
                system_prompt=SAFETY_JUDGE_SYSTEM_PROMPT.strip(),
                user_prompt=user_prompt,
            )
            safe, reason = parse_judge_response(response)
            return {**turn, "judge_safe": safe, "judge_reason": reason, "judge_provider": "gemini"}
        except Exception as e:
            return {**turn, "judge_safe": None, "judge_reason": None, "judge_error": str(e), "judge_provider": "gemini"}

    # Run concurrent judging
    tasks = [_judge_one(job) for job in jobs]
    total = 0

    with tqdm(total=len(jobs), desc="Judging safety", unit="turn") as pbar:
        async for result in run_concurrent_iter(tasks, config.max_concurrent):
            write_jsonl(config.scored_path, [result])
            total += 1
            pbar.update(1)

    print(f"Wrote {total} judged records to {config.scored_path}")
    return config.scored_path


__all__ = ["JudgeConfig", "judge_safety"]
