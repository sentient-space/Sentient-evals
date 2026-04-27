from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


CaptureMode = Literal[
    "native_stream_json",
    "native_json",
    "native_trajectory",
    "session_jsonl",
    "session_scrape",
    "stdout_fallback",
]
SupportLevel = Literal["guaranteed", "best_effort", "none"]
TrajectoryConfidence = Literal["native", "normalized", "degraded", "none"]


@dataclass(frozen=True)
class AdapterCapabilities:
    capture_mode: CaptureMode = "stdout_fallback"
    tool_call_support: SupportLevel = "none"
    metrics_support: SupportLevel = "none"
    trajectory_confidence: TrajectoryConfidence = "degraded"
    notes: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


STDOUT_ONLY_CAPABILITIES = AdapterCapabilities(
    capture_mode="stdout_fallback",
    tool_call_support="none",
    metrics_support="none",
    trajectory_confidence="degraded",
    notes="Adapter emits only stdout/stderr fallback transcript unless a subclass parser succeeds.",
)
