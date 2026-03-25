from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from .base import JudgeClient, JudgeResponse

_TIMEOUT_SECONDS = 60.0
_MAX_RETRIES = 2
_OPENAI_BASE_URL = "https://api.openai.com/v1"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_SUPPORTED_PREFIXES = {
    "anthropic",
    "gemini",
    "google",
    "openai",
    "openai-compatible",
    "openrouter",
}


def _normalize_provider(provider: str | None) -> str | None:
    value = str(provider or "").strip().lower()
    if not value:
        return None
    if value == "gemini":
        return "google"
    return value


def _infer_provider(model: str) -> str:
    lowered = model.strip().lower()
    if "/" in lowered:
        prefix, _rest = lowered.split("/", 1)
        normalized = _normalize_provider(prefix)
        if normalized in _SUPPORTED_PREFIXES:
            return normalized
    if lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith("gemini"):
        return "google"
    if lowered.startswith(("gpt-", "o1", "o3", "o4", "o5")):
        return "openai"
    return "openai"


def _resolve_provider_and_model(provider: str | None, model: str) -> tuple[str, str]:
    raw_model = str(model or "").strip()
    if not raw_model:
        raise RuntimeError("Judge model is required")

    normalized_provider = _normalize_provider(provider)
    if normalized_provider:
        return normalized_provider, raw_model

    inferred_provider = _infer_provider(raw_model)
    if "/" in raw_model:
        prefix, remainder = raw_model.split("/", 1)
        if _normalize_provider(prefix) == inferred_provider and remainder.strip():
            return inferred_provider, remainder.strip()
    return inferred_provider, raw_model


def _extract_openai_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "".join(parts).strip()
    return ""


def _extract_anthropic_text(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if text:
                parts.append(str(text))
    return "".join(parts).strip()


def _extract_google_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    first = candidates[0]
    if not isinstance(first, dict):
        return ""
    content = first.get("content")
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    texts: list[str] = []
    for item in parts:
        if isinstance(item, dict):
            text = item.get("text")
            if text:
                texts.append(str(text))
    return "".join(texts).strip()


@dataclass(frozen=True)
class DirectJudgeClient(JudgeClient):
    api_key: str | None = None
    provider: str | None = None
    base_url: str | None = None

    async def score(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> JudgeResponse:
        provider, resolved_model = _resolve_provider_and_model(self.provider, model)

        if provider == "openai":
            raw = await self._call_openai(
                model=resolved_model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return JudgeResponse(content=_extract_openai_text(raw), raw=raw)

        if provider == "openrouter":
            raw = await self._call_openrouter(
                model=resolved_model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return JudgeResponse(content=_extract_openai_text(raw), raw=raw)

        if provider == "openai-compatible":
            raw = await self._call_openai_compatible(
                model=resolved_model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return JudgeResponse(content=_extract_openai_text(raw), raw=raw)

        if provider == "anthropic":
            raw = await self._call_anthropic(
                model=resolved_model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return JudgeResponse(content=_extract_anthropic_text(raw), raw=raw)

        if provider == "google":
            raw = await self._call_google(
                model=resolved_model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return JudgeResponse(content=_extract_google_text(raw), raw=raw)

        raise RuntimeError(f"Unsupported judge provider '{provider}' for model '{model}'")

    async def _call_openai(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        base_url = (self.base_url or os.getenv("OPENAI_BASE_URL") or _OPENAI_BASE_URL).rstrip("/")
        return await self._post_json(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._require_api_key('OPENAI_API_KEY')}",
                "Content-Type": "application/json",
            },
            body={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

    async def _call_openrouter(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        base_url = (self.base_url or os.getenv("OPENROUTER_BASE_URL") or _OPENROUTER_BASE_URL).rstrip("/")
        headers = {
            "Authorization": f"Bearer {self._require_api_key('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        }
        app_name = str(os.getenv("OPENROUTER_APP_NAME", "")).strip()
        http_referer = str(os.getenv("OPENROUTER_HTTP_REFERER", "")).strip()
        if app_name:
            headers["X-Title"] = app_name
        if http_referer:
            headers["HTTP-Referer"] = http_referer
        return await self._post_json(
            f"{base_url}/chat/completions",
            headers=headers,
            body={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

    async def _call_openai_compatible(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        base_url = (
            self.base_url
            or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("LLM_BASE_URL")
        )
        if not base_url:
            raise RuntimeError(
                "OpenAI-compatible judge models require base_url or one of "
                "OPENAI_COMPATIBLE_BASE_URL / OPENAI_BASE_URL / LLM_BASE_URL"
            )
        return await self._post_json(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._require_api_key('OPENAI_API_KEY')}",
                "Content-Type": "application/json",
            },
            body={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

    async def _call_anthropic(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        return await self._post_json(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self._require_api_key("ANTHROPIC_API_KEY"),
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            body={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

    async def _call_google(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        api_key = self._require_api_key("GEMINI_API_KEY")
        return await self._post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            body={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            },
        )

    def _require_api_key(self, env_name: str) -> str:
        api_key = str(self.api_key or "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key for direct judge client. Expected {env_name}.")
        return api_key

    async def _post_json(self, url: str, *, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        try:
            import httpx  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install sentient-evals[llm] to use judge-based graders") from exc

        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    response = await client.post(url, headers=headers, json=body)
                    response.raise_for_status()
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise RuntimeError("Judge provider returned a non-object JSON response")
                    return payload
                except (httpx.HTTPError, ValueError, RuntimeError) as exc:
                    last_error = exc
                    if attempt < _MAX_RETRIES:
                        await asyncio.sleep(1.5**attempt)
                        continue
        raise RuntimeError(f"Judge API call failed after {_MAX_RETRIES + 1} attempts: {last_error}")
