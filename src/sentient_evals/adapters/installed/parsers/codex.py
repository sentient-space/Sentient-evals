from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .common import ParseResult, message_event, tool_call_event, utcnow
from ....models import TranscriptEvent

logger = logging.getLogger(__name__)


def _parse_output_blob(raw: Any) -> tuple[str | None, dict[str, Any] | None]:
    if raw is None:
        return None, None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw, None
    else:
        parsed = raw
    if isinstance(parsed, dict):
        output = parsed.get("output")
        if output is None and parsed:
            output = json.dumps(parsed, ensure_ascii=False)
        metadata = parsed.get("metadata")
        return output, metadata if isinstance(metadata, dict) else None
    return str(parsed), None


def _find_session_file(base: Path) -> Path | None:
    sessions = base / "sessions"
    if not sessions.exists():
        return None
    session_files = list(sessions.rglob("*.jsonl"))
    if not session_files:
        return None
    
    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except Exception:
            return 0.0

    session_files.sort(key=lambda p: (_mtime(p), p.as_posix()))
    return session_files[-1]


def parse_codex_session(trial_env_logs_agent_dir: Path, *, instruction: str) -> ParseResult | None:
    session_file = _find_session_file(trial_env_logs_agent_dir)
    if session_file is None:
        return None

    raw_events: list[dict[str, Any]] = []
    for line in session_file.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            raw_events.append(obj)

    if not raw_events:
        return None

    events: list[TranscriptEvent] = [message_event(role="user", content=instruction, t=utcnow())]
    metrics: dict[str, float] = {}

    for e in raw_events:
        et = e.get("type")
        payload = e.get("payload") if isinstance(e.get("payload"), dict) else {}
        if et == "message":
            role = payload.get("role") or "assistant"
            text = payload.get("text") or payload.get("content") or ""
            reasoning = payload.get("reasoning")
            ev = message_event(role=role if role in {"user", "assistant", "system"} else "assistant", content=text)
            if isinstance(reasoning, str) and reasoning.strip():
                ev.reasoning_content = reasoning
            usage = payload.get("usage")
            if isinstance(usage, dict):
                if usage.get("prompt_tokens") is not None:
                    ev.metrics = ev.metrics or {}
                    ev.metrics["prompt_tokens"] = float(usage.get("prompt_tokens") or 0)
                if usage.get("completion_tokens") is not None:
                    ev.metrics = ev.metrics or {}
                    ev.metrics["completion_tokens"] = float(usage.get("completion_tokens") or 0)
                if usage.get("cached_tokens") is not None:
                    ev.metrics = ev.metrics or {}
                    ev.metrics["cached_tokens"] = float(usage.get("cached_tokens") or 0)
                if usage.get("cost_usd") is not None:
                    ev.metrics = ev.metrics or {}
                    ev.metrics["cost_usd"] = float(usage.get("cost_usd") or 0.0)
            events.append(ev)
            continue

        if et == "tool_call":
            call_id = payload.get("call_id") or payload.get("id") or ""
            tool_name = payload.get("tool_name") or payload.get("name") or ""
            raw_args = payload.get("arguments")
            args = raw_args if isinstance(raw_args, dict) else {"value": raw_args}
            output, meta = _parse_output_blob(payload.get("output"))
            tc = tool_call_event(name=str(tool_name or "tool"), args=args, observation=output)
            tc.tool_call.id = str(call_id) if call_id else None  # type: ignore[union-attr]
            if meta:
                tc.extra = {"tool_metadata": meta}
            events.append(tc)
            continue

    return ParseResult(events=events, metrics=metrics, extra={"source": "codex-session", "session_file": session_file.name})

