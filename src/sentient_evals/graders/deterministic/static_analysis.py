from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ...artifacts import TrialArtifacts
from ...env import ToolExecutor
from ...models import GraderResult, Severity, Task, TranscriptEvent


def _truncate(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


@dataclass(frozen=True)
class StaticAnalysisSpec:
    name: str
    cmd: str
    timeout_s: float | None = None


@dataclass(frozen=True)
class StaticAnalysisGrader:
    name: str = "static_analysis"
    checks: Sequence[StaticAnalysisSpec] = ()

    async def grade(
        self, *, task: Task, transcript: Sequence[TranscriptEvent], outcome, artifacts: TrialArtifacts
    ) -> GraderResult:
        return GraderResult(
            name=self.name,
            score=0.0,
            passed=False,
            severity=Severity.error,
            details={"error": "StaticAnalysisGrader requires env; use grade_with_env"},
        )

    async def grade_with_env(
        self,
        *,
        task: Task,
        transcript: Sequence[TranscriptEvent],
        outcome,
        artifacts: TrialArtifacts,
        env: ToolExecutor,
    ) -> GraderResult:
        if not self.checks:
            return GraderResult(
                name=self.name,
                score=0.0,
                passed=False,
                severity=Severity.error,
                details={"error": "StaticAnalysisGrader requires checks"},
            )

        results = []
        passed_all = True
        for check in self.checks:
            res = await env.exec(check.cmd, timeout_s=check.timeout_s)
            ok = res.exit_code == 0
            passed_all = passed_all and ok
            results.append(
                {
                    "name": check.name,
                    "cmd": check.cmd,
                    "exit_code": res.exit_code,
                    "duration_ms": res.duration_ms,
                    "stdout": _truncate(res.stdout),
                    "stderr": _truncate(res.stderr),
                    "passed": ok,
                }
            )

        artifacts.verifier().write_json("static_analysis.json", results)
        score = sum(1 for r in results if r["passed"]) / float(len(results))
        return GraderResult(
            name=self.name,
            score=score,
            passed=passed_all,
            severity=Severity.info if passed_all else Severity.error,
            details={"checks": results},
        )
