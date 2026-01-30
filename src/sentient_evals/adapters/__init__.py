from __future__ import annotations

import json
from typing import Protocol

from ..artifacts import TrialArtifacts
from ..env import ToolExecutor
from ..models import Outcome, Task, TranscriptEvent
from .installed.base import BaseInstalledAdapter, ExecCommand


class AgentAdapter(Protocol):
    name: str

    async def run(
        self,
        task: Task,
        *,
        instruction: str | None,
        seed: int,
        env: ToolExecutor,
        artifacts: TrialArtifacts,
    ) -> tuple[list[TranscriptEvent], Outcome]:
        ...


class WorkflowAdapter(Protocol):
    name: str

    async def run(
        self,
        task: Task,
        *,
        instruction: str | None,
        seed: int,
        env: ToolExecutor,
        artifacts: TrialArtifacts,
    ) -> tuple[list[TranscriptEvent], Outcome]:
        ...


class ExampleInstalledAdapter(BaseInstalledAdapter):
    name = "example_installed"
    version = "0.1.0"

    @property
    def install_template_path(self):
        raise FileNotFoundError("ExampleInstalledAdapter does not support install")

    async def install(self, env: ToolExecutor, artifacts: TrialArtifacts) -> None:
        return None

    def create_run_commands(self, instruction: str, *, task: Task, seed: int) -> list[ExecCommand]:
        return [ExecCommand(cmd=f"echo {json.dumps({'instruction': instruction})}")]


class WorkflowStubAdapter:
    name = "workflow_stub"
    version = "0.1.0"

    async def run(
        self,
        task: Task,
        *,
        instruction: str | None,
        seed: int,
        env: ToolExecutor,
        artifacts: TrialArtifacts,
    ) -> tuple[list[TranscriptEvent], Outcome]:
        transcript = [
            TranscriptEvent(kind="message", role="user", content=instruction or json.dumps(task.input)),
            TranscriptEvent(kind="metric", metrics={"workflow_steps": 1.0}, content="route=default"),
            TranscriptEvent(kind="message", role="assistant", content="(workflow stub)"),
        ]
        return transcript, Outcome(summary="ok", data={"answer": "workflow_stub"})
