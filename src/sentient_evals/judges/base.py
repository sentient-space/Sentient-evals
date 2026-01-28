from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class JudgeResponse:
    content: str
    raw: Any


class JudgeClient(Protocol):
    async def score(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> JudgeResponse:
        ...
