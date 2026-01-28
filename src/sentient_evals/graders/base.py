from __future__ import annotations

from typing import Any, Protocol, Sequence

from ..artifacts import TrialArtifacts
from ..env import ToolExecutor
from ..models import GraderResult, Task, TranscriptEvent


class Grader(Protocol):
    name: str

    async def grade(
        self,
        *,
        task: Task,
        transcript: Sequence[TranscriptEvent],
        outcome: Any,
        artifacts: TrialArtifacts,
    ) -> GraderResult:
        ...


class EnvironmentGrader(Protocol):
    name: str

    async def grade_with_env(
        self,
        *,
        task: Task,
        transcript: Sequence[TranscriptEvent],
        outcome: Any,
        artifacts: TrialArtifacts,
        env: ToolExecutor,
    ) -> GraderResult:
        ...


def supports_env_grading(grader: object) -> bool:
    return hasattr(grader, "grade_with_env")
