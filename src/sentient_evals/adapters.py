from __future__ import annotations

from typing import Protocol

from .env import ToolExecutor
from .models import Outcome, Task, TranscriptEvent


class AgentAdapter(Protocol):
    name: str

    async def run(
        self, task: Task, *, seed: int, env: ToolExecutor
    ) -> tuple[list[TranscriptEvent], Outcome]:
        ...

