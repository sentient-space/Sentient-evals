from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ...artifacts import TrialArtifacts
from ...models import GraderResult, Severity, Task, TranscriptEvent


@dataclass(frozen=True)
class ExactMatchGrader:
    name: str = "exact_match"
    field: str = "answer"
    expected: str = ""

    async def grade(
        self, *, task: Task, transcript: Sequence[TranscriptEvent], outcome: Any, artifacts: TrialArtifacts
    ) -> GraderResult:
        actual = ""
        if isinstance(outcome, dict):
            actual = str(outcome.get(self.field, ""))
        else:
            actual = str(getattr(outcome, self.field, "") if outcome is not None else "")
        passed = actual == self.expected
        return GraderResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            severity=Severity.info if passed else Severity.error,
            details={"expected": self.expected, "actual": actual, "field": self.field},
        )
