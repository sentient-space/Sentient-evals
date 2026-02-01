from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ...artifacts import TrialArtifacts
from ...models import GraderResult, Severity, Task, TranscriptEvent


@dataclass(frozen=True)
class HumanReviewSpec:
    sample_id: str | None = None
    notes: str | None = None
    include_transcript: bool = True
    include_outcome: bool = True


@dataclass(frozen=True)
class HumanReviewGrader:
    name: str = "human_review"
    spec: HumanReviewSpec = HumanReviewSpec()

    async def grade(
        self, *, task: Task, transcript: Sequence[TranscriptEvent], outcome: Any, artifacts: TrialArtifacts
    ) -> GraderResult:
        packet = {
            "sample_id": self.spec.sample_id,
            "notes": self.spec.notes,
            "task": task.model_dump(),
            "transcript": [e.model_dump() for e in transcript] if self.spec.include_transcript else None,
            "outcome": outcome if self.spec.include_outcome else None,
        }
        artifacts.judge().write_json("human_review_packet.json", packet)
        return GraderResult(
            name=self.name,
            score=1.0,
            passed=True,
            severity=Severity.info,
            details={"recorded": True, "sample_id": self.spec.sample_id},
        )
