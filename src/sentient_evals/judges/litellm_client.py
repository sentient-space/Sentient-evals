from __future__ import annotations

from dataclasses import dataclass

from .base import JudgeClient, JudgeResponse


@dataclass(frozen=True)
class LiteLLMJudgeClient(JudgeClient):
    api_key: str | None = None

    async def score(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> JudgeResponse:
        try:
            from litellm import acompletion  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install sentient-evals[llm] to use LiteLLMJudgeClient") from exc

        resp = await acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.api_key,
        )
        try:
            raw = resp.model_dump()
        except Exception:  # pragma: no cover
            raw = {"raw": str(resp)}
        content = (resp.choices[0].message.content or "").strip()
        return JudgeResponse(content=content, raw=raw)
