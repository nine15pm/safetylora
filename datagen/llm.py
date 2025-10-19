"""Async SDK helpers for data generation."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any, Dict

from google import genai
from google.genai import types
from openai import AsyncOpenAI


_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


OPENAI_PROVIDERS: Dict[str, Dict[str, str]] = {
    "openai": {"env": "OPENAI_API_KEY", "model": "gpt-4o-mini"},
    "moonshot": {
        "env": "MOONSHOT_API_KEY",
        "base_url": "https://api.moonshot.ai/v1",
        "model": "kimi-k2-turbo-preview",
    },
    "deepseek": {
        "env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "grok": {
        "env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "model": "grok-4-fast-non-reasoning",
    },
}


GEMINI_PROVIDER: Dict[str, str] = {
    "env": "GEMINI_API_KEY",
    "model": "gemini-2.5-flash",
}


class OpenAICompletion:
    """Async wrapper for OpenAI-compatible text generations."""

    def __init__(self, provider: str, model: str | None = None, **default_params: Any) -> None:
        name = provider.lower()
        if name not in OPENAI_PROVIDERS:
            raise ValueError(f"Unknown OpenAI provider '{provider}'.")

        settings = OPENAI_PROVIDERS[name]
        api_key_env = settings.get("env")
        api_key = os.getenv(api_key_env) if api_key_env else None
        if api_key_env and not api_key:
            raise RuntimeError(
                f"Environment variable '{api_key_env}' is required for provider '{provider}'."
            )

        client_kwargs: Dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        base_url = settings.get("base_url")
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**client_kwargs)
        self._model = model or settings.get("model")
        if not self._model:
            raise ValueError(f"No model provided for provider '{provider}'.")
        self._defaults = default_params

    async def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        **overrides: Any,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        params = {**self._defaults, **overrides}
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **params,
        )
        choice = response.choices[0].message
        content = getattr(choice, "content", None)
        if isinstance(content, list):
            text = "".join(
                (part.get("text") if isinstance(part, dict) else getattr(part, "text", "")) or ""
                for part in content
            )
        else:
            text = content or getattr(choice, "text", None)
        return str(text or "")


class GeminiCompletion:
    """Async wrapper for Gemini content generation."""

    def __init__(self, model: str | None = None, **default_params: Any) -> None:
        api_key_env = GEMINI_PROVIDER.get("env")
        api_key = os.getenv(api_key_env) if api_key_env else None
        if api_key_env and not api_key:
            raise RuntimeError(
                f"Environment variable '{api_key_env}' is required for Gemini."
            )

        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._model = model or GEMINI_PROVIDER.get("model")
        if not self._model:
            raise ValueError("No Gemini model provided.")
        self._defaults = default_params
        self._closed = False

    async def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        **overrides: Any,
    ) -> str:
        params = {**self._defaults, **overrides}

        # Build config with system_instruction if provided
        config_kwargs = {}
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        config_kwargs.update(params)

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = getattr(response, "text", None) or getattr(response, "output_text", None)
        return str(text or "")

    async def aclose(self) -> None:
        """Close underlying async client session if still open."""
        if self._closed:
            return

        aio_iface = getattr(self._client, "aio", None)
        to_close = []
        if aio_iface:
            close_callable = getattr(aio_iface, "close", None)
            if callable(close_callable):
                to_close.append(close_callable)

            session = getattr(aio_iface, "session", None)
            if session:
                session_close = getattr(session, "close", None)
                if callable(session_close):
                    to_close.append(session_close)

        for closer in to_close:
            result = closer()
            if inspect.isawaitable(result):
                await result

        self._closed = True


__all__ = ["OpenAICompletion", "GeminiCompletion", "OPENAI_PROVIDERS", "GEMINI_PROVIDER"]
