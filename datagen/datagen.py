"""Clean async data generation for user and assistant turns."""

import asyncio
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from datagen.llm import OpenAICompletion
from datagen.prompts import (
    USER_TURN_SYSTEM_PROMPT,
    USER_TURN_USER_PROMPT,
    ASSISTANT_TURN_SYSTEM_PROMPT,
    ASSISTANT_TURN_USER_PROMPT,
)


# Core async pattern: semaphore + as_completed for incremental results
async def run_concurrent_iter(tasks: list, max_concurrent: int):
    """Run tasks with concurrency limit, yielding results as they complete."""
    sem = asyncio.Semaphore(max_concurrent)

    async def _run_with_sem(task):
        async with sem:
            return await task

    for coro in asyncio.as_completed([_run_with_sem(task) for task in tasks]):
        yield await coro


# Basic I/O utilities
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


# Configs
@dataclass
class UserTurnConfig:
    provider: str
    temperature: float
    max_tokens: int
    num_messages: int = 1
    max_concurrent: int = 4
    resume: bool = True
    intent_seeds_file: str = "intentseeds.jsonl"
    user_turns_file: str = "userturns.jsonl"

    @property
    def intent_seeds_path(self) -> Path:
        return Path(__file__).parent / self.intent_seeds_file

    @property
    def user_turns_path(self) -> Path:
        return Path(__file__).parent / self.user_turns_file


@dataclass
class AssistantTurnConfig:
    provider: str
    temperature: float
    max_tokens: int
    max_concurrent: int = 4
    resume: bool = True
    system_prompts_file: str = "sysprompts.jsonl"
    user_turns_file: str = "userturns.jsonl"
    assistant_turns_file: str = "assistantturns.jsonl"

    @property
    def system_prompts_path(self) -> Path:
        return Path(__file__).parent / self.system_prompts_file

    @property
    def user_turns_path(self) -> Path:
        return Path(__file__).parent / self.user_turns_file

    @property
    def assistant_turns_path(self) -> Path:
        return Path(__file__).parent / self.assistant_turns_file


# User turn generation
async def generate_user_turns(config: UserTurnConfig) -> Path:
    """Generate all user turns from intent seeds."""
    # Load seeds
    seeds = read_jsonl(config.intent_seeds_path)
    if not seeds:
        print(f"No seeds found at {config.intent_seeds_path}")
        return config.user_turns_path

    # Filter already completed (if resume enabled)
    if config.resume:
        completed_ids = {row["seed_id"] for row in read_jsonl(config.user_turns_path)
                        if row.get("provider") == config.provider and "seed_id" in row}
        pending = [s for s in seeds if s["seed_id"] not in completed_ids]
    else:
        pending = seeds

    if not pending:
        print(f"All {len(seeds)} seeds already completed")
        return config.user_turns_path

    # Create model
    model = OpenAICompletion(
        provider=config.provider,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    # Run all generations concurrently with incremental writes
    async def _generate_one(seed):
        user_prompt = USER_TURN_USER_PROMPT.format(
            num_messages=config.num_messages,
            input_context=f"INTENT: {seed['intent']}\nTYPE: {seed['type']}\nCATEGORY: {seed['category']}"
        )
        try:
            response = await model.generate(
                system_prompt=USER_TURN_SYSTEM_PROMPT.strip(),
                user_prompt=user_prompt
            )
            # Parse JSONL response
            messages = [json.loads(line.strip()) for line in response.strip().split("\n") if line.strip()]
            # Build output records
            return [{
                "turn_id": str(uuid.uuid4()),
                "seed_id": seed["seed_id"],
                "intent": seed["intent"],
                "category": seed["category"],
                "provider": config.provider,
                "type": msg.get("type", seed["type"]),
                "user_msg": msg["user_msg"],
            } for msg in messages]
        except Exception as e:
            return [{
                "seed_id": seed["seed_id"],
                "intent": seed["intent"],
                "category": seed["category"],
                "provider": config.provider,
                "error": str(e),
            }]

    tasks = [_generate_one(seed) for seed in pending]
    total_records = 0

    with tqdm(total=len(pending), desc=f"User turns ({config.provider})", unit="seed") as pbar:
        async for results in run_concurrent_iter(tasks, config.max_concurrent):
            # Flatten and write each result as it completes
            records = results if isinstance(results, list) else [results]
            write_jsonl(config.user_turns_path, records)
            total_records += len(records)
            pbar.update(1)

    print(f"Wrote {total_records} records to {config.user_turns_path}")
    return config.user_turns_path


# Assistant turn generation
async def generate_assistant_turns(config: AssistantTurnConfig) -> Path:
    """Generate all assistant turns for user messages."""
    # Load inputs
    system_prompts = read_jsonl(config.system_prompts_path)
    user_turns = read_jsonl(config.user_turns_path)

    if not system_prompts or not user_turns:
        print("Missing system prompts or user turns")
        return config.assistant_turns_path

    # Build jobs: one per (user_turn, system_prompt) pair
    jobs = []
    for turn in user_turns:
        if turn.get("error") or not turn.get("user_msg"):
            continue

        for sp in system_prompts:
            jobs.append({
                "user_msg": turn["user_msg"],
                "system_prompt": sp,
                "metadata": {
                    "turn_id": turn.get("turn_id"),
                    "seed_id": turn.get("seed_id"),
                    "category": turn.get("category"),
                    "system_prompt_id": sp.get("id") or sp.get("name"),
                    "provider": config.provider,
                    "user_msg": turn["user_msg"],
                }
            })

    if not jobs:
        print("No jobs to process")
        return config.assistant_turns_path

    # Filter completed if resuming (use turn_id + system_prompt_id)
    if config.resume:
        completed = {
            (row["turn_id"], row["system_prompt_id"])
            for row in read_jsonl(config.assistant_turns_path)
            if row.get("provider") == config.provider
            and "turn_id" in row
            and "system_prompt_id" in row
        }
        jobs = [
            j for j in jobs
            if (j["metadata"]["turn_id"], j["metadata"]["system_prompt_id"]) not in completed
        ]

    if not jobs:
        print("All jobs already completed")
        return config.assistant_turns_path

    # Create model
    model = OpenAICompletion(
        provider=config.provider,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    # Run all generations concurrently with incremental writes
    async def _generate_one(job):
        input_parts = [
            json.dumps({"system_prompt": job["system_prompt"]["prompt"]}, ensure_ascii=False),
            json.dumps({"user_prompt": job["user_msg"]}, ensure_ascii=False),
        ]
        user_prompt = ASSISTANT_TURN_USER_PROMPT.format(input_context="\n".join(input_parts))
        try:
            response = await model.generate(
                system_prompt=ASSISTANT_TURN_SYSTEM_PROMPT.strip(),
                user_prompt=user_prompt
            )
            return {**job["metadata"], "assistant_msg": response.strip()}
        except Exception as e:
            return {**job["metadata"], "assistant_msg": "", "error": str(e)}

    tasks = [_generate_one(job) for job in jobs]
    total_records = 0

    with tqdm(total=len(jobs), desc=f"Assistant turns ({config.provider})", unit="turn") as pbar:
        async for result in run_concurrent_iter(tasks, config.max_concurrent):
            write_jsonl(config.assistant_turns_path, [result])
            total_records += 1
            pbar.update(1)

    print(f"Wrote {total_records} records to {config.assistant_turns_path}")
    return config.assistant_turns_path
