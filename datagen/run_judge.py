import asyncio

from datagen.judge import JudgeConfig, judge_safety


async def main() -> None:
    cfg = JudgeConfig(
        temperature=0.6,
        max_tokens=4096,
        max_concurrent=15,
        resume=False,
        safety_policy_file="safetypolicy.jsonl",
        assistant_turns_file="assistantturns_errors.jsonl",
        scored_file="assistantturns_errors_scored.jsonl",
    )
    path = await judge_safety(cfg)
    print(f"Wrote judge scores to {path}")


if __name__ == "__main__":
    asyncio.run(main())
