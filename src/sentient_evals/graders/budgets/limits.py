from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ...artifacts import TrialArtifacts
from ...models import GraderResult, Severity, Task, TranscriptEvent


def _get_metric(outcome: Any, key: str) -> float | None:
    if isinstance(outcome, dict):
        if key in outcome:
            return _to_float(outcome.get(key))
        metrics = outcome.get("metrics")
        if isinstance(metrics, dict):
            return _to_float(metrics.get(key))
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


@dataclass(frozen=True)
class BudgetGrader:
    name: str = "budget"
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_latency_ms: int | None = None

    async def grade(
        self, *, task: Task, transcript: Sequence[TranscriptEvent], outcome: Any, artifacts: TrialArtifacts
    ) -> GraderResult:
        metrics = {
            "tokens": _get_metric(outcome, "total_tokens"),
            "cost_usd": _get_metric(outcome, "total_cost_usd"),
            "latency_ms": _get_metric(outcome, "latency_ms"),
        }

        failures = []
        if self.max_tokens is not None and metrics["tokens"] is not None:
            if metrics["tokens"] > self.max_tokens:
                failures.append({"metric": "total_tokens", "value": metrics["tokens"]})
        if self.max_cost_usd is not None and metrics["cost_usd"] is not None:
            if metrics["cost_usd"] > self.max_cost_usd:
                failures.append({"metric": "total_cost_usd", "value": metrics["cost_usd"]})
        if self.max_latency_ms is not None and metrics["latency_ms"] is not None:
            if metrics["latency_ms"] > self.max_latency_ms:
                failures.append({"metric": "latency_ms", "value": metrics["latency_ms"]})

        missing = []
        if self.max_tokens is not None and metrics["tokens"] is None:
            missing.append("total_tokens")
        if self.max_cost_usd is not None and metrics["cost_usd"] is None:
            missing.append("total_cost_usd")
        if self.max_latency_ms is not None and metrics["latency_ms"] is None:
            missing.append("latency_ms")

        passed = not failures and not missing
        severity = Severity.info if passed else Severity.warning

        artifacts.verifier().write_json(
            "budget.json",
            {"limits": {"tokens": self.max_tokens, "cost_usd": self.max_cost_usd, "latency_ms": self.max_latency_ms}, "metrics": metrics, "failures": failures, "missing": missing},
        )

        return GraderResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            severity=severity,
            details={"failures": failures, "missing": missing, "metrics": metrics},
        )
