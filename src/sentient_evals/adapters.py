from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .env import ToolExecutor
from .models import Outcome, Task, TranscriptEvent


class AgentAdapter(Protocol):
    name: str

    async def run(
        self, task: Task, *, seed: int, env: ToolExecutor
    ) -> tuple[list[TranscriptEvent], Outcome]:
        ...


class WorkflowAdapter(Protocol):
    name: str

    async def run(
        self, task: Task, *, seed: int, env: ToolExecutor
    ) -> tuple[list[TranscriptEvent], Outcome]:
        ...


@dataclass(frozen=True)
class ExecCommand:
    cmd: str
    timeout_s: float | None = None


class BaseInstalledAdapter:
    name = "installed_base"
    version = "unknown"

    def __init__(self) -> None:
        self._installed = False

    async def install(self, env: ToolExecutor) -> None:
        return None

    def build_instruction(self, task: Task) -> str:
        if task.input:
            return json.dumps(task.input)
        return task.id

    def run_commands(self, instruction: str, *, seed: int, task: Task) -> Sequence[ExecCommand]:
        raise NotImplementedError

    async def parse_logs(
        self,
        *,
        task: Task,
        instruction: str,
        env: ToolExecutor,
        results: Sequence[Any],
    ) -> list[TranscriptEvent]:
        output = ""
        if results:
            last = results[-1]
            stdout = getattr(last, "stdout", "") if last else ""
            stderr = getattr(last, "stderr", "") if last else ""
            output = "\n".join([part for part in [stdout, stderr] if part])
        return [
            TranscriptEvent(kind="message", role="user", content=instruction),
            TranscriptEvent(kind="message", role="assistant", content=output.strip()),
        ]

    def build_outcome(self, transcript: Sequence[TranscriptEvent]) -> Outcome:
        answer = ""
        for ev in reversed(transcript):
            if ev.kind == "message" and ev.role == "assistant":
                answer = ev.content or ""
                break
        return Outcome(summary="ok", data={"answer": answer})

    async def run(
        self, task: Task, *, seed: int, env: ToolExecutor
    ) -> tuple[list[TranscriptEvent], Outcome]:
        if not self._installed:
            await self.install(env)
            self._installed = True
        instruction = self.build_instruction(task)
        commands = self.run_commands(instruction, seed=seed, task=task)
        results = []
        for cmd in commands:
            results.append(await env.exec(cmd.cmd, timeout_s=cmd.timeout_s))
        transcript = await self.parse_logs(
            task=task, instruction=instruction, env=env, results=results
        )
        return transcript, self.build_outcome(transcript)


class ExampleInstalledAdapter(BaseInstalledAdapter):
    name = "example_installed"
    version = "0.1.0"

    def run_commands(self, instruction: str, *, seed: int, task: Task) -> Sequence[ExecCommand]:
        return [ExecCommand(cmd=f"echo {json.dumps({'instruction': instruction})}")]


class WorkflowStubAdapter:
    name = "workflow_stub"
    version = "0.1.0"

    async def run(
        self, task: Task, *, seed: int, env: ToolExecutor
    ) -> tuple[list[TranscriptEvent], Outcome]:
        transcript = [
            TranscriptEvent(kind="message", role="user", content=json.dumps(task.input)),
            TranscriptEvent(kind="metric", metrics={"workflow_steps": 1.0}, content="route=default"),
            TranscriptEvent(kind="message", role="assistant", content="(workflow stub)"),
        ]
        return transcript, Outcome(summary="ok", data={"answer": "workflow_stub"})

