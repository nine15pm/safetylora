from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from openai import OpenAI


_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

_CONFIG = {
    "openai": {"api_key_env": "OPENAI_API_KEY"},
    "moonshot": {
        "api_key_env": "MOONSHOT_API_KEY",
        "base_url": "https://api.moonshot.ai/v1",
        "model": "kimi-k2-turbo-preview",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "grok": {
        "api_key_env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "model": "grok-4",
    },
}


class LLM:
    def __init__(self, provider: str, model: str | None = None, **defaults: Any) -> None:
        cfg = _CONFIG.get(provider.lower())
        if not cfg:
            raise ValueError(f"Unsupported provider '{provider}'.")

        api_key_env = cfg.get("api_key_env")
        kwargs = {k: cfg[k] for k in ("base_url",) if k in cfg}
        self._defaults = {**cfg.get("defaults", {}), **defaults}
        if api_key_env:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Environment variable '{api_key_env}' is required for provider '{provider}'."
                )
            kwargs["api_key"] = api_key

        self._client = OpenAI(**kwargs)
        self._model = model or cfg.get("model")
        if not self._model:
            raise ValueError(f"No model provided and provider '{provider}' has no default.")

    def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        params = {**self._defaults, **kwargs}
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **params,
        )
        message = resp.choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, list):
            content = "".join(
                (part.get("text") if isinstance(part, dict) else getattr(part, "text", str(part))) or ""
                for part in content
            )
        if not content:
            reasoning = getattr(message, "reasoning_content", None)
            if isinstance(reasoning, list):
                content = "".join(
                    (part.get("text") if isinstance(part, dict) else getattr(part, "text", str(part))) or ""
                    for part in reasoning
                )
            elif reasoning:
                content = reasoning
        if not content:
            content = getattr(message, "text", None)
        if not content and isinstance(message, dict):
            content = message.get("text") or message.get("content")
        return str(content or "")
