from __future__ import annotations

from pathlib import Path

from datagen.judge import JudgeConfig, run_judge


def main() -> None:
    cfg = JudgeConfig(
        assistant_turns_path=Path(__file__).with_name("assistantturns_test.jsonl"),
        scored_path=Path(__file__).with_name("assistantturns_test_scored.jsonl"),
        resume=False,
        provider="deepseek",
    )
    path = run_judge(cfg)
    print(f"Wrote judge scores to {path}")


if __name__ == "__main__":
    main()
