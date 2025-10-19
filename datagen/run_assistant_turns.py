import asyncio
from pathlib import Path

from datagen.datagen import AssistantTurnConfig, generate_assistant_turns

async def main() -> None:
    cfg = AssistantTurnConfig(
        provider="grok",
        temperature=0.7,
        max_tokens=4096,
        max_concurrent=15,
        resume=True,
        system_prompts_file="sysprompts.jsonl",
        user_turns_file="userturns.jsonl",
        assistant_turns_file="assistantturns.jsonl",
    )
    path = await generate_assistant_turns(cfg)
    print(f"Wrote assistant turns to {path}")


if __name__ == "__main__":
    asyncio.run(main())
