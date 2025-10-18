from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from datagen.llm import LLM
from datagen.prompts import (
    ASSISTANT_TURN_SYSTEM_PROMPT,
    ASSISTANT_TURN_USER_PROMPT,
    USER_TURN_SYSTEM_PROMPT,
    USER_TURN_USER_PROMPT,
)

@dataclass(frozen=True)
class UserTurnConfig:
    providers: tuple[str, ...] = ("moonshot", "deepseek", "grok")
    temperature: float = 0.7
    max_tokens: int = 4096
    num_messages: int = 1
    resume: bool = True
    user_turns_path: Path = field(
        default_factory=lambda: Path(__file__).with_name("userturns.jsonl")
    )
    intent_seeds_path: Path = field(
        default_factory=lambda: Path(__file__).with_name("intentseeds.jsonl")
    )


@dataclass
class UserTurnRecord:
    seed_id: str | None
    intent: str | None
    category: str | None
    provider: str
    type: str | None
    user_msg: str | None = None
    error: str | None = None

    @classmethod
    def from_seed(
        cls,
        seed: Dict[str, str],
        provider: str,
        *,
        user_msg: str | None = None,
        type_override: str | None = None,
        error: str | None = None,
    ) -> "UserTurnRecord":
        return cls(
            seed_id=seed.get("seed_id"),
            intent=seed.get("intent"),
            category=seed.get("category"),
            provider=provider,
            type=type_override or seed.get("type"),
            user_msg=user_msg,
            error=error,
        )

    def to_json(self) -> str:
        payload = {
            "seed_id": self.seed_id,
            "intent": self.intent,
            "category": self.category,
            "provider": self.provider,
            "type": self.type,
        }
        if self.user_msg is not None:
            payload["user_msg"] = self.user_msg
        if self.error is not None:
            payload["error"] = self.error
        return json.dumps(payload, ensure_ascii=False)


@dataclass(frozen=True)
class AssistantTurnConfig:
    provider: str = "moonshot"
    temperature: float = 0.2
    max_tokens: int = 1024
    resume: bool = True
    system_prompts_path: Path = field(
        default_factory=lambda: Path(__file__).with_name("sysprompts.jsonl")
    )
    safety_policy_path: Path = field(
        default_factory=lambda: Path(__file__).with_name("safetypolicy.jsonl")
    )
    user_turns_path: Path = field(
        default_factory=lambda: Path(__file__).with_name("userturns.jsonl")
    )
    assistant_turns_path: Path = field(
        default_factory=lambda: Path(__file__).with_name("assistantturns.jsonl")
    )


@dataclass
class AssistantTurnRecord:
    turn_id: str | None
    seed_id: str | None
    category: str | None
    system_prompt_id: str
    provider: str
    user_msg: str
    assistant_msg: str

    def to_json(self) -> str:
        payload = {
            "turn_id": self.turn_id,
            "seed_id": self.seed_id,
            "category": self.category,
            "system_prompt_id": self.system_prompt_id,
            "provider": self.provider,
            "user_msg": self.user_msg,
            "assistant_msg": self.assistant_msg,
        }
        if payload["turn_id"] is None:
            del payload["turn_id"]
        return json.dumps(payload, ensure_ascii=False)


def _parse_jsonl(payload: str) -> List[Dict[str, str]]:
    """Parse a JSONL string into a list of objects, with a simple JSON array fallback."""
    records: List[Dict[str, str]] = []
    lines = [line.strip() for line in payload.splitlines() if line.strip()]
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON line: {line}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object per line, found {type(obj)}.")
        records.append(obj)
    if records:
        return records

    payload = payload.strip()
    if not payload:
        return []

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse provider response: {payload[:1000]}") from exc

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        if not all(isinstance(item, dict) for item in parsed):
            raise ValueError("JSON array must contain objects.")
        return parsed
    raise ValueError(f"Unexpected JSON payload type: {type(parsed)}")


def _build_user_prompt(seed: Dict[str, str], num_messages: int = 1) -> str:
    intent = seed["intent"]
    intent_type = seed["type"]
    category = seed.get("category", "unknown")
    input_context = "\n".join(
        (
            f"INTENT: {intent}",
            f"TYPE: {intent_type}",
            f"CATEGORY: {category}",
        )
    )
    return USER_TURN_USER_PROMPT.format(
        num_messages=num_messages, input_context=input_context
    ).strip()


def _call_provider(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    llm = LLM(
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)


def _read_jsonl(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_intent_seeds(path: Path) -> List[Dict[str, str]]:
    return _read_jsonl(path)


def _load_completed_records(path: Path) -> Dict[str, set[str]]:
    if not path.exists():
        return {}

    completed: Dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            seed_id = record.get("seed_id")
            provider = record.get("provider")
            if seed_id and provider:
                completed.setdefault(provider, set()).add(seed_id)
    return completed


def _load_completed_assistant_turns(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            turn_id = record.get("turn_id")
            seed_ref = turn_id or record.get("seed_id") or record.get("user_msg")
            system_prompt_id = record.get("system_prompt_id")
            if system_prompt_id:
                completed.add((str(seed_ref or ""), str(system_prompt_id)))
    return completed


def _load_system_prompts(path: Path) -> List[Dict[str, str]]:
    return _read_jsonl(path)


def _load_safety_policies(path: Path) -> Dict[str, Dict[str, str]]:
    policies: Dict[str, Dict[str, str]] = {}
    for row in _read_jsonl(path):
        category = row.get("category")
        policy = row.get("policy")
        if category and isinstance(policy, dict):
            policies[category] = policy
    return policies


def _build_assistant_user_prompt(
    *,
    system_prompt: Dict[str, str],
    user_turn: Dict[str, str],
    safety_policy: Dict[str, str] | None,
) -> str:
    prompt_text = (system_prompt.get("prompt") or "").strip()
    system_payload = {"system_prompt": prompt_text}

    user_msg = (user_turn.get("user_msg") or "").strip()
    user_payload = {"user_prompt": user_msg}

    policy_payload: Dict[str, str] = {}
    if safety_policy:
        policy_payload.update(safety_policy)
    category = user_turn.get("category")
    if category:
        policy_payload.setdefault("category", category)

    lines = [
        json.dumps(system_payload, ensure_ascii=False),
        json.dumps(user_payload, ensure_ascii=False),
        json.dumps({"safety_policy": policy_payload}, ensure_ascii=False),
    ]
    return ASSISTANT_TURN_USER_PROMPT.format(
        input_context="\n".join(lines)
    ).strip()


def generate_user_turns(config: UserTurnConfig | None = None) -> Path:
    cfg = config or UserTurnConfig()
    seeds = _load_intent_seeds(path=cfg.intent_seeds_path)
    if not cfg.providers:
        return cfg.user_turns_path

    output_path = cfg.user_turns_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_by_provider = (
        _load_completed_records(output_path) if cfg.resume else {}
    )
    system_prompt = USER_TURN_SYSTEM_PROMPT.strip()

    with output_path.open("a", encoding="utf-8") as handle:
        for provider in cfg.providers:
            provider_completed = completed_by_provider.setdefault(provider, set())
            progress = tqdm(
                total=len(seeds),
                initial=len(provider_completed) if cfg.resume else 0,
                desc=f"Generating user turns ({provider})",
                unit="seed",
            )
            try:
                for seed in seeds:
                    seed_id = seed.get("seed_id")
                    if cfg.resume and seed_id and seed_id in provider_completed:
                        continue

                    user_prompt = _build_user_prompt(seed, cfg.num_messages)
                    error_message: str | None = None
                    messages: List[Dict[str, str]] = []
                    try:
                        content = _call_provider(
                            provider,
                            system_prompt,
                            user_prompt,
                            temperature=cfg.temperature,
                            max_tokens=cfg.max_tokens,
                        )
                    except Exception as exc:  # noqa: BLE001
                        error_message = str(exc)
                    else:
                        try:
                            messages = _parse_jsonl(content)
                        except ValueError as exc:
                            error_message = str(exc)

                    if error_message is not None:
                        record = UserTurnRecord.from_seed(
                            seed,
                            provider,
                            error=error_message,
                        )
                        handle.write(record.to_json() + "\n")
                    else:
                        for message in messages:
                            record = UserTurnRecord.from_seed(
                                seed,
                                provider,
                                user_msg=(message.get("user_msg") or "").strip(),
                                type_override=message.get("type"),
                            )
                            handle.write(record.to_json() + "\n")

                    if seed_id:
                        provider_completed.add(seed_id)
                    progress.update(1)
            finally:
                progress.close()
    return output_path


def generate_assistant_turns(config: AssistantTurnConfig | None = None) -> Path:
    cfg = config or AssistantTurnConfig()
    system_prompts = _load_system_prompts(cfg.system_prompts_path)
    user_turns = _read_jsonl(cfg.user_turns_path)
    if not system_prompts or not user_turns:
        return cfg.assistant_turns_path

    policies = _load_safety_policies(cfg.safety_policy_path)
    completed = (
        _load_completed_assistant_turns(cfg.assistant_turns_path)
        if cfg.resume
        else set()
    )

    output_path = cfg.assistant_turns_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt_template = ASSISTANT_TURN_SYSTEM_PROMPT.strip()
    model = LLM(
        provider=cfg.provider,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )

    with output_path.open("a", encoding="utf-8") as handle:
        for turn in tqdm(user_turns, desc="Generating assistant turns", unit="turn"):
            if turn.get("error"):
                continue
            user_msg = (turn.get("user_msg") or "").strip()
            if not user_msg:
                continue
            category = turn.get("category")
            policy = policies.get(category or "")
            turn_id = turn.get("turn_id")
            seed_ref = turn_id or turn.get("seed_id") or user_msg

            for system_prompt in system_prompts:
                system_prompt_id = system_prompt.get("id") or system_prompt.get("name")
                prompt_text = (system_prompt.get("prompt") or "").strip()
                if not system_prompt_id or not prompt_text:
                    continue
                key = (str(seed_ref or ""), str(system_prompt_id))
                if cfg.resume and key in completed:
                    continue

                user_prompt = _build_assistant_user_prompt(
                    system_prompt=system_prompt,
                    user_turn=turn,
                    safety_policy=policy,
                )
                assistant_msg = model.generate(
                    system_prompt=system_prompt_template,
                    user_prompt=user_prompt,
                )
                record = AssistantTurnRecord(
                    turn_id=turn_id,
                    seed_id=turn.get("seed_id"),
                    category=category,
                    system_prompt_id=str(system_prompt_id),
                    provider=cfg.provider,
                    user_msg=user_msg,
                    assistant_msg=assistant_msg.strip(),
                )
                handle.write(record.to_json() + "\n")
                completed.add(key)

    return output_path


def main() -> None:
    path = generate_user_turns()
    print(f"Appended user turns to {path}")


if __name__ == "__main__":
    main()
