import asyncio

from datagen.safetyrewrite import RewriteConfig, rewrite_unsafe


async def main() -> None:
    cfg = RewriteConfig(
        temperature=0.7,
        max_tokens=4096,
        max_concurrent=15,
        resume=True,
        safety_policy_file="safetypolicy.jsonl",
        system_prompts_file="sysprompts.jsonl",
        unsafe_turns_file="assistantturns_errors_scored.jsonl",
        rewritten_file="assistantturns_errors_rewritten.jsonl",
    )
    path = await rewrite_unsafe(cfg)
    print(f"Wrote rewritten responses to {path}")


if __name__ == "__main__":
    asyncio.run(main())
