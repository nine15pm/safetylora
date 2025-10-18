from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm
from datagen.datagen import _load_safety_policies, _read_jsonl
from datagen.llm import LLM
from datagen.prompts import SAFETY_JUDGE_SYSTEM_PROMPT, SAFETY_JUDGE_USER_PROMPT


@dataclass(frozen=True)
class JudgeConfig:
    provider: str = "gemini"
    temperature: float = 0.6
    max_tokens: int = 4096
    resume: bool = True
    assistant_turns_path: Path = field(
        default_factory=lambda: Path(__file__).with_name("assistantturns_test.jsonl")
    )
    safety_policy_path: Path = field(
        default_factory=lambda: Path(__file__).with_name("safetypolicy.jsonl")
    )
    scored_path: Path = field(
        default_factory=lambda: Path(__file__).with_name("assistantturns_scored.jsonl")
    )


def _parse_judge_output(raw: str) -> Tuple[int, str]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty judge output.")
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Could not parse judge JSON: {raw}") from None
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Judge output must be a JSON object.")

    safe_val = payload.get("safe")
    if isinstance(safe_val, str) and safe_val.isdigit():
        safe_val = int(safe_val, 10)
    if isinstance(safe_val, bool):
        safe_val = int(safe_val)
    if not isinstance(safe_val, int) or safe_val not in (0, 1):
        raise ValueError(f"Invalid safe value: {safe_val!r}")

    reason = payload.get("reason")
    reason_str = str(reason).strip() if reason is not None else ""
    if safe_val == 0 and not reason_str:
        raise ValueError("Unsafe judgement must include a reason.")
    if safe_val == 1 and not reason_str:
        reason_str = "none"
    return safe_val, reason_str


def _scored_keys(path: Path) -> set[Tuple[str, str]]:
    keys: set[Tuple[str, str]] = set()
    if not path.exists():
        return keys
    for row in _read_jsonl(path):
        if row.get("judge_safe") not in (0, 1):
            continue
        seed_ref = row.get("seed_id") or row.get("user_msg") or ""
        system_prompt_id = row.get("system_prompt_id")
        if not system_prompt_id:
            continue
        keys.add((str(seed_ref), str(system_prompt_id)))
    return keys


def run_judge(config: JudgeConfig | None = None) -> Path:
    cfg = config or JudgeConfig()
    assistant_turns = _read_jsonl(cfg.assistant_turns_path)
    if not assistant_turns:
        return cfg.scored_path

    policies = _load_safety_policies(cfg.safety_policy_path)
    skip_keys = _scored_keys(cfg.scored_path) if cfg.resume else set()

    judge = LLM(
        provider=cfg.provider,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    system_prompt = SAFETY_JUDGE_SYSTEM_PROMPT.strip()
    cfg.scored_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.scored_path.open("a", encoding="utf-8") as handle:
        for row in tqdm(assistant_turns, desc="Scoring responses", unit="turn"):
            user_msg = (row.get("user_msg") or "").strip()
            assistant_msg = (row.get("assistant_msg") or "").strip()
            system_prompt_id = row.get("system_prompt_id")
            if not user_msg or not assistant_msg or not system_prompt_id:
                continue

            seed_ref = str(row.get("seed_id") or row.get("user_msg") or "")
            key = (seed_ref, str(system_prompt_id))
            if cfg.resume and key in skip_keys:
                continue

            category = str(row.get("category") or "")
            policy: Dict | None = policies.get(category) if category else None
            policy_blob = {"category": category} if category else {}
            if policy:
                policy_blob.update(policy)

            user_prompt = SAFETY_JUDGE_USER_PROMPT.format(
                safety_policy=json.dumps(policy_blob or {"category": "unknown"}, ensure_ascii=False),
                user_prompt=f"USER PROMPT:\n{user_msg}",
                assistant_response=f"ASSISTANT RESPONSE:\n{assistant_msg}",
            )

            record = dict(row)
            record["judge_provider"] = cfg.provider

            try:
                raw = judge.generate(system_prompt=system_prompt, user_prompt=user_prompt)
                score, reason = _parse_judge_output(raw)
            except Exception as exc:  # noqa: BLE001
                record["judge_safe"] = None
                record["judge_reason"] = None
                record["judge_error"] = str(exc)
            else:
                record["judge_safe"] = score
                record["judge_reason"] = reason
                if cfg.resume:
                    skip_keys.add(key)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return cfg.scored_path


__all__ = ["JudgeConfig", "run_judge"]
