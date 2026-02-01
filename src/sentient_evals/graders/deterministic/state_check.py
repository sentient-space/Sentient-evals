from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from ...artifacts import TrialArtifacts
from ...models import GraderResult, Severity, Task, TranscriptEvent


def _match_operator(expected: dict[str, Any], actual: Any) -> tuple[bool, str]:
    if "$eq" in expected:
        return actual == expected["$eq"], "$eq"
    if "$ne" in expected:
        return actual != expected["$ne"], "$ne"
    if "$gt" in expected:
        return actual > expected["$gt"], "$gt"
    if "$gte" in expected:
        return actual >= expected["$gte"], "$gte"
    if "$lt" in expected:
        return actual < expected["$lt"], "$lt"
    if "$lte" in expected:
        return actual <= expected["$lte"], "$lte"
    if "$in" in expected:
        return actual in expected["$in"], "$in"
    if "$contains" in expected:
        return expected["$contains"] in actual, "$contains"
    if "$regex" in expected:
        return re.search(str(expected["$regex"]), str(actual)) is not None, "$regex"
    return False, "unknown"


def _match_expected(expected: Any, actual: Any) -> tuple[bool, list[str]]:
    mismatches: list[str] = []

    if isinstance(expected, dict):
        op_keys = [k for k in expected.keys() if k.startswith("$")]
        if op_keys:
            ok, op = _match_operator(expected, actual)
            if not ok:
                mismatches.append(f"operator {op} failed (actual={actual})")
            return ok, mismatches
        if not isinstance(actual, dict):
            mismatches.append("actual is not a dict")
            return False, mismatches
        for key, val in expected.items():
            ok, sub = _match_expected(val, actual.get(key))
            if not ok:
                mismatches.append(f"{key}: {', '.join(sub)}")
        return not mismatches, mismatches

    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False, ["actual is not a list"]
        if len(expected) != len(actual):
            return False, [f"length {len(actual)} != {len(expected)}"]
        for idx, (e, a) in enumerate(zip(expected, actual)):
            ok, sub = _match_expected(e, a)
            if not ok:
                mismatches.append(f"[{idx}]: {', '.join(sub)}")
        return not mismatches, mismatches

    return (expected == actual), ([] if expected == actual else [f"{actual} != {expected}"])


@dataclass(frozen=True)
class StateCheckGrader:
    name: str = "state_check"
    expect: dict[str, Any] | None = None
    outcome_field: str | None = None

    async def grade(
        self, *, task: Task, transcript: Sequence[TranscriptEvent], outcome: Any, artifacts: TrialArtifacts
    ) -> GraderResult:
        if self.expect is None:
            return GraderResult(
                name=self.name,
                score=0.0,
                passed=False,
                severity=Severity.error,
                details={"error": "StateCheckGrader requires expect"},
            )

        subject = outcome
        if self.outcome_field and isinstance(outcome, dict):
            subject = outcome.get(self.outcome_field)

        ok, mismatches = _match_expected(self.expect, subject)
        return GraderResult(
            name=self.name,
            score=1.0 if ok else 0.0,
            passed=ok,
            severity=Severity.info if ok else Severity.error,
            details={
                "expect": self.expect,
                "outcome_field": self.outcome_field,
                "mismatches": mismatches,
            },
        )
