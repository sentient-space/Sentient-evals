from __future__ import annotations

from typing import Any, Awaitable, Callable

from sentient_evals.adapters.frameworks.common import ToolExecutorRecorder, build_instruction_payload
from sentient_evals.env import ToolExecutor
from sentient_evals.models import Outcome, Task, TranscriptEvent


class AutoGenAdapter:
    name = "autogen"

    def __init__(
        self,
        *,
        runner: Callable[[str, ToolExecutorRecorder], Awaitable[Any]],
    ) -> None:
        self.runner = runner

    async def run(
        self, task: Task, *, seed: int, env: ToolExecutor
    ) -> tuple[list[TranscriptEvent], Outcome]:
        try:
            import autogen  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Install sentient-evals[autogen] to use AutoGenAdapter"
            ) from exc

        instruction = build_instruction_payload(task.input)
        transcript: list[TranscriptEvent] = [
            TranscriptEvent(kind="message", role="user", content=instruction)
        ]
        recorder = ToolExecutorRecorder(env, transcript)
        result = await self.runner(instruction, recorder)
        transcript.append(
            TranscriptEvent(kind="message", role="assistant", content=str(result))
        )
        return transcript, Outcome(summary="ok", data={"answer": result})
