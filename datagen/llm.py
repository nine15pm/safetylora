"""Minimal LLM wrapper that normalizes provider-specific SDKs.

Gemini uses the official google-genai SDK. All other providers are assumed to
expose an OpenAI-compatible chat API and are driven through the OpenAI client.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from openai import OpenAI


_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {"env": "OPENAI_API_KEY", "kind": "openai"},
    "gemini": {"env": "GEMINI_API_KEY", "kind": "gemini", "model": "gemini-2.5-flash"},
    "moonshot": {
        "env": "MOONSHOT_API_KEY",
        "base_url": "https://api.moonshot.ai/v1",
        "model": "kimi-k2-turbo-preview",
        "kind": "openai",
    },
    "deepseek": {
        "env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "kind": "openai",
    },
    "grok": {
        "env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "model": "grok-4-fast-non-reasoning",
        "kind": "openai",
    },
}


class LLM:
    def __init__(self, provider: str, model: str | None = None, **defaults: Any) -> None:
        name = provider.lower()
        cfg = _PROVIDERS.get(name)
        if cfg is None:
            raise ValueError(f"Unsupported provider '{provider}'.")

        api_key_env = cfg.get("env")
        api_key = os.environ.get(api_key_env) if api_key_env else None
        if api_key_env and not api_key:
            raise RuntimeError(
                f"Environment variable '{api_key_env}' is required for provider '{provider}'."
            )

        self._defaults = {**cfg.get("defaults", {}), **defaults}
        self._model = model or cfg.get("model")
        if not self._model:
            raise ValueError(f"No model provided and provider '{provider}' has no default.")

        kind = cfg.get("kind", "openai")
        if kind == "gemini":
            self._adapter = _GeminiAdapter(model=self._model, api_key=api_key)
        else:
            client_kwargs: dict[str, Any] = {}
            if api_key:
                client_kwargs["api_key"] = api_key
            if "base_url" in cfg:
                client_kwargs["base_url"] = cfg["base_url"]
            self._adapter = _OpenAIAdapter(model=self._model, client_kwargs=client_kwargs)

    def generate(self, system_prompt: str, user_prompt: str, **overrides: Any) -> str:
        params = {**self._defaults, **overrides}
        return self._adapter.generate(system_prompt=system_prompt, user_prompt=user_prompt, params=params)


class _Adapter:
    def generate(self, system_prompt: str, user_prompt: str, params: dict[str, Any]) -> str:
        raise NotImplementedError


class _GeminiAdapter(_Adapter):
    def __init__(self, model: str, api_key: str | None) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._model = model

    def generate(self, system_prompt: str, user_prompt: str, params: dict[str, Any]) -> str:
        from google.genai import types as genai_types

        params = dict(params)
        config = params.pop("config", None)
        categories = (
            genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        )
        safety_settings = [
            genai_types.SafetySetting(
                category=category,
                threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
            )
            for category in categories
        ]
        system_instruction = system_prompt or None
        if config is None and system_prompt:
            config = genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
                safety_settings=safety_settings,
            )
        elif config is None:
            config = genai_types.GenerateContentConfig(safety_settings=safety_settings)
        elif isinstance(config, genai_types.GenerateContentConfig):
            if system_instruction is not None:
                config.system_instruction = system_instruction
            config.safety_settings = safety_settings
        else:
            config_dict = dict(config)
            if system_instruction is not None:
                config_dict["system_instruction"] = system_instruction
            config_dict["safety_settings"] = safety_settings
            config = genai_types.GenerateContentConfig(**config_dict)
        params.pop("generation_config", None)
        overrides = {
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "max_tokens": "max_output_tokens",
            "max_output_tokens": "max_output_tokens",
            "candidate_count": "candidate_count",
            "stop_sequences": "stop_sequences",
            "stop": "stop_sequences",
            "frequency_penalty": "frequency_penalty",
            "presence_penalty": "presence_penalty",
            "response_logprobs": "response_logprobs",
            "logprobs": "logprobs",
            "seed": "seed",
            "response_mime_type": "response_mime_type",
            "response_schema": "response_schema",
        }
        for key, target in overrides.items():
            value = params.pop(key, None)
            if value is None:
                continue
            if target == "stop_sequences" and isinstance(value, str):
                value = [value]
            setattr(config, target, value)
        response = self._client.models.generate_content(
            model=self._model,
            contents=user_prompt,
            config=config,
            **params,
        )
        text = getattr(response, "output_text", None) or getattr(response, "text", None)
        if text:
            return str(text)
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None)
            if parts:
                return "".join(getattr(part, "text", "") for part in parts)
            finish_reason = getattr(candidates[0], "finish_reason", None)
            if finish_reason:
                print(f"[Gemini] finish_reason={finish_reason}")
        else:
            print("[Gemini] finish_reason=none (no candidates)")
        return ""


class _OpenAIAdapter(_Adapter):
    def __init__(self, model: str, client_kwargs: dict[str, Any]) -> None:
        self._client = OpenAI(**client_kwargs)
        self._model = model

    def generate(self, system_prompt: str, user_prompt: str, params: dict[str, Any]) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **params,
        )
        message = response.choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, list):
            content = "".join(
                (part.get("text") if isinstance(part, dict) else getattr(part, "text", "")) or ""
                for part in content
            )
        if not content:
            content = getattr(message, "text", None)
        if not content and isinstance(message, dict):
            content = message.get("content") or message.get("text")
        return str(content or "")
