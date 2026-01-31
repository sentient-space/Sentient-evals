from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ....models import ToolCall, TranscriptEvent

logger = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def safe_parse_timestamp(value: Any) -> datetime | None:
    """
    Best-effort parse timestamps from agent logs.
    Accepts ISO strings or unix seconds/ms.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Heuristic: values > 1e12 are probably ms.
        seconds = float(value) / 1000.0 if float(value) > 1e12 else float(value)
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            # Support trailing Z
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s)
        except Exception:
            return None
    return None


def iter_json_lines(text: str) -> Iterable[dict[str, Any]]:
    """
    Parse a blob as JSONL/NDJSON. Lines that don't parse are skipped.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj


@dataclass(frozen=True)
class ParseResult:
    events: list[TranscriptEvent]
    metrics: dict[str, float]
    tool_definitions: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None


def try_parse_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    return list(iter_json_lines(text))


def fallback_stdout_transcript(*, instruction: str, stdout: str) -> list[TranscriptEvent]:
    return [
        TranscriptEvent(kind="message", role="user", content=instruction),
        TranscriptEvent(kind="message", role="assistant", content=stdout.strip()),
    ]


def tool_call_event(*, name: str, args: dict[str, Any] | None = None, observation: Any | None = None) -> TranscriptEvent:
    return TranscriptEvent(
        kind="tool_call",
        role="assistant",
        content=None,
        tool_call=ToolCall(name=name, args=args or {}),
        observation=observation,
    )


def message_event(*, role: str, content: Any, t: datetime | None = None) -> TranscriptEvent:
    ev = TranscriptEvent(kind="message", role=role, content=_as_str(content))
    if t is not None:
        ev.t = t
    return ev

