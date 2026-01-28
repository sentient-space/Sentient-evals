from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ...artifacts import TrialArtifacts
from ...models import GraderResult, Severity, Task, TranscriptEvent
from .llm import LLMJudgeGrader


@dataclass(frozen=True)
class MultiJudgeAggregation(str):
    majority = "majority"
    unanimous = "unanimous"
    average_score = "average_score"


@dataclass(frozen=True)
class MultiLLMJudgeGrader:
    name: str = "multi_llm_judge"
    judges: Sequence[LLMJudgeGrader] = ()
    aggregation: MultiJudgeAggregation = MultiJudgeAggregation.majority

    async def grade(
        self, *, task: Task, transcript: Sequence[TranscriptEvent], outcome, artifacts: TrialArtifacts
    ) -> GraderResult:
        if not self.judges:
            artifacts.judge().write_json("error.json", {"error": "No judges configured"})
            return GraderResult(
                name=self.name,
                score=0.0,
                passed=False,
                severity=Severity.error,
                details={"error": "No judges configured"},
            )

        results: list[GraderResult] = []
        for i, j in enumerate(self.judges):
            j_art = artifacts.scoped(f"judge_{i}")
            results.append(await j.grade(task=task, transcript=transcript, outcome=outcome, artifacts=j_art))

        scores = [r.score for r in results]
        passes = [r.passed for r in results]
        agreement_rate = sum(1 for p in passes if p) / float(len(passes)) if passes else 0.0

        if self.aggregation == MultiJudgeAggregation.unanimous:
            passed = all(passes)
            score = float(sum(scores)) / float(len(scores))
        elif self.aggregation == MultiJudgeAggregation.average_score:
            score = float(sum(scores)) / float(len(scores))
            passed = score >= 0.5
        else:
            passed = sum(1 for p in passes if p) > (len(passes) // 2)
            score = float(sum(scores)) / float(len(scores))

        artifacts.judge().write_json(
            "aggregation.json",
            {
                "aggregation": self.aggregation,
                "passed": passed,
                "score": score,
                "agreement_rate": agreement_rate,
                "judges": [{"name": r.name, "passed": r.passed, "score": r.score} for r in results],
            },
        )
        return GraderResult(
            name=self.name,
            score=score,
            passed=passed,
            severity=Severity.info if passed else Severity.error,
            details={
                "aggregation": self.aggregation,
                "agreement_rate": agreement_rate,
                "judges": [{"name": r.name, "passed": r.passed, "score": r.score} for r in results],
            },
        )
