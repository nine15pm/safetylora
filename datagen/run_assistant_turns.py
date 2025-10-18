from __future__ import annotations

from pathlib import Path

from datagen.datagen import AssistantTurnConfig, generate_assistant_turns


def main() -> None:
    cfg = AssistantTurnConfig(
        provider="grok",
        user_turns_path=Path(__file__).with_name("userturns_test.jsonl"),
        assistant_turns_path=Path(__file__).with_name("assistantturns_test.jsonl"),
        resume=False,
    )
    path = generate_assistant_turns(cfg)
    print(f"Wrote assistant turns to {path}")


if __name__ == "__main__":
    main()
