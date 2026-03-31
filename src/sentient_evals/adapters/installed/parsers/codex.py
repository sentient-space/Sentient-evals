from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .common import ParseResult, message_event, safe_parse_timestamp, tool_call_event, utcnow
from ....models import TranscriptEvent

logger = logging.getLogger(__name__)


def _append_unique(parts: list[str], value: Any) -> None:
    if value is None:
        return
    text = ""
    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            text = json.dumps(value, ensure_ascii=False).strip()
        except Exception:
            text = str(value).strip()
    if text and text not in parts:
        parts.append(text)


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


def _parse_arguments(raw: Any) -> tuple[dict[str, Any], Any | None]:
    if isinstance(raw, dict):
        return raw, None
    if raw in (None, ""):
        return {}, None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"value": raw}, raw
        if isinstance(parsed, dict):
            return parsed, None
        return {"value": parsed}, raw
    return {"value": raw}, raw


def _extract_message_content(raw: Any) -> tuple[str, str | None]:
    if isinstance(raw, str):
        return raw.strip(), None
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                _append_unique(text_parts, item)
                continue
            if not isinstance(item, dict):
                _append_unique(text_parts, item)
                continue
            item_type = str(item.get("type") or "").strip().lower()
            if item_type in {"output_text", "input_text", "text"}:
                _append_unique(text_parts, item.get("text"))
                continue
            if item_type in {"reasoning", "reasoning_text", "thinking", "summary_text"}:
                _append_unique(reasoning_parts, item.get("text"))
                continue
            if item_type in {"output_image", "image"}:
                _append_unique(text_parts, item.get("image_url") or item.get("url") or "[image]")
                continue
            _append_unique(text_parts, item.get("text") or item)
    elif raw is not None:
        _append_unique(text_parts, raw)
    text = "\n\n".join(text_parts)
    reasoning = "\n\n".join(reasoning_parts)
    return text, (reasoning or None)


def _extract_reasoning_summary(raw: Any) -> str | None:
    parts: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                _append_unique(parts, item.get("text") or item.get("summary") or item.get("content") or item)
            else:
                _append_unique(parts, item)
    else:
        _append_unique(parts, raw)
    joined = "\n\n".join(parts).strip()
    return joined or None


def _merge_metrics(metrics: dict[str, float], payload: dict[str, Any] | None) -> None:
    if not isinstance(payload, dict):
        return
    prompt_tokens = payload.get("input_tokens")
    completion_tokens = payload.get("output_tokens")
    cached_tokens = payload.get("cached_input_tokens")
    for key, raw in (
        ("prompt_tokens", prompt_tokens),
        ("completion_tokens", completion_tokens),
        ("cached_tokens", cached_tokens),
    ):
        if raw is None:
            continue
        try:
            value = float(raw)
        except Exception:
            continue
        metrics[key] = max(metrics.get(key, 0.0), value)


def _step_metrics_from_usage(payload: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(payload, dict):
        return None
    prompt_tokens = payload.get("input_tokens")
    completion_tokens = payload.get("output_tokens")
    cached_tokens = payload.get("cached_input_tokens")
    step_metrics: dict[str, float] = {}
    for key, raw in (
        ("prompt_tokens", prompt_tokens),
        ("completion_tokens", completion_tokens),
        ("cached_tokens", cached_tokens),
    ):
        if raw is None:
            continue
        try:
            step_metrics[key] = float(raw)
        except Exception:
            continue
    return step_metrics or None


def _attach_usage_to_last_turn_event(
    events: list[TranscriptEvent],
    turn_event_indexes: list[int],
    usage_payload: dict[str, Any] | None,
) -> None:
    step_metrics = _step_metrics_from_usage(usage_payload)
    if not step_metrics or not turn_event_indexes:
        return
    target = events[turn_event_indexes[-1]]
    merged = dict(target.metrics or {})
    merged.update(step_metrics)
    target.metrics = merged


def _finalize_parse(
    *,
    instruction: str,
    parsed_events: list[TranscriptEvent],
    metrics: dict[str, float],
    extra: dict[str, Any],
) -> ParseResult | None:
    meaningful_events = [
        event
        for event in parsed_events
        if (event.kind == "message" and event.role == "assistant" and ((event.content or "").strip() or event.reasoning_content))
        or event.kind == "tool_call"
    ]
    if not meaningful_events:
        return None
    events = [message_event(role="user", content=instruction, t=utcnow())]
    events.extend(parsed_events)
    return ParseResult(events=events, metrics=metrics, extra=extra or None)


def _parse_codex_session_events(
    raw_events: list[dict[str, Any]],
    *,
    instruction: str,
    extra: dict[str, Any] | None = None,
) -> ParseResult | None:
    events: list[TranscriptEvent] = []
    metrics: dict[str, float] = {}
    pending_tool_calls: dict[str, TranscriptEvent] = {}
    pending_reasoning: list[str] = []
    turn_event_indexes: list[int] = []

    def _flush_reasoning(target: TranscriptEvent | None = None) -> None:
        nonlocal pending_reasoning
        if not pending_reasoning:
            return
        reasoning_text = "\n\n".join(part for part in pending_reasoning if part).strip()
        pending_reasoning = []
        if not reasoning_text:
            return
        if target is not None:
            target.reasoning_content = reasoning_text
            return
        events.append(
            TranscriptEvent(
                kind="message",
                role="assistant",
                content="",
                reasoning_content=reasoning_text,
            )
        )
        turn_event_indexes.append(len(events) - 1)

    for raw_event in raw_events:
        event_type = raw_event.get("type")
        payload = raw_event.get("payload") if isinstance(raw_event.get("payload"), dict) else {}
        timestamp = safe_parse_timestamp(raw_event.get("timestamp")) or utcnow()

        if event_type == "message":
            role = payload.get("role") or "assistant"
            if role not in {"assistant", "system"}:
                continue
            text, inline_reasoning = _extract_message_content(payload.get("content") or payload.get("text") or payload.get("content"))
            event = message_event(
                role=role if role in {"assistant", "system"} else "assistant",
                content=text,
                t=timestamp,
            )
            if inline_reasoning:
                event.reasoning_content = inline_reasoning
            _merge_metrics(metrics, payload.get("usage"))
            if pending_reasoning and not event.reasoning_content:
                _flush_reasoning(event)
            events.append(event)
            turn_event_indexes.append(len(events) - 1)
            continue

        if event_type == "tool_call":
            call_id = payload.get("call_id") or payload.get("id") or ""
            tool_name = payload.get("tool_name") or payload.get("name") or ""
            args, raw_args = _parse_arguments(payload.get("arguments"))
            output, meta = _parse_output_blob(payload.get("output"))
            tool_event = tool_call_event(name=str(tool_name or "tool"), args=args, observation=output)
            tool_event.t = timestamp
            tool_event.tool_call.id = str(call_id) if call_id else None  # type: ignore[union-attr]
            extra_payload: dict[str, Any] = {}
            if meta:
                extra_payload["tool_metadata"] = meta
            if raw_args is not None:
                extra_payload["raw_arguments"] = raw_args
            if pending_reasoning:
                _flush_reasoning(tool_event)
            if extra_payload:
                tool_event.extra = extra_payload
            events.append(tool_event)
            turn_event_indexes.append(len(events) - 1)
            if call_id:
                pending_tool_calls[str(call_id)] = tool_event
            continue

        if event_type == "event_msg":
            payload_type = payload.get("type")
            if payload_type == "token_count":
                info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
                total = info.get("total_token_usage") if isinstance(info.get("total_token_usage"), dict) else {}
                last = info.get("last_token_usage") if isinstance(info.get("last_token_usage"), dict) else {}
                _merge_metrics(metrics, total)
                _attach_usage_to_last_turn_event(events, turn_event_indexes, last or total)
                turn_event_indexes = []
            continue

        if event_type != "response_item":
            continue

        payload_type = payload.get("type")
        if payload_type == "message":
            role = str(payload.get("role") or "assistant").strip().lower()
            if role != "assistant":
                continue
            text, inline_reasoning = _extract_message_content(payload.get("content"))
            event = message_event(role="assistant", content=text, t=timestamp)
            if inline_reasoning:
                event.reasoning_content = inline_reasoning
            if isinstance(payload.get("phase"), str):
                event.extra = {"phase": payload.get("phase")}
            if pending_reasoning and not event.reasoning_content:
                _flush_reasoning(event)
            events.append(event)
            turn_event_indexes.append(len(events) - 1)
            continue

        if payload_type == "function_call":
            call_id = str(payload.get("call_id") or payload.get("id") or "").strip()
            tool_name = str(payload.get("name") or "tool").strip() or "tool"
            args, raw_args = _parse_arguments(payload.get("arguments"))
            tool_event = tool_call_event(name=tool_name, args=args)
            tool_event.t = timestamp
            tool_event.tool_call.id = call_id or None  # type: ignore[union-attr]
            extra_payload: dict[str, Any] = {}
            if raw_args is not None:
                extra_payload["raw_arguments"] = raw_args
            if pending_reasoning:
                _flush_reasoning(tool_event)
            if extra_payload:
                tool_event.extra = extra_payload
            events.append(tool_event)
            turn_event_indexes.append(len(events) - 1)
            if call_id:
                pending_tool_calls[call_id] = tool_event
            continue

        if payload_type == "function_call_output":
            call_id = str(payload.get("call_id") or payload.get("id") or "").strip()
            output, meta = _parse_output_blob(payload.get("output"))
            existing = pending_tool_calls.get(call_id)
            if existing is not None:
                existing.observation = output
                if meta:
                    existing.extra = {**(existing.extra or {}), "tool_metadata": meta}
                continue
            tool_event = tool_call_event(name="tool", args={}, observation=output)
            tool_event.t = timestamp
            tool_event.tool_call.id = call_id or None  # type: ignore[union-attr]
            if meta:
                tool_event.extra = {"tool_metadata": meta}
            if pending_reasoning:
                _flush_reasoning(tool_event)
            events.append(tool_event)
            turn_event_indexes.append(len(events) - 1)
            continue

        if payload_type == "reasoning":
            reasoning_text = _extract_reasoning_summary(payload.get("summary") or payload.get("content"))
            if reasoning_text:
                pending_reasoning.append(reasoning_text)
            continue

        if payload_type == "web_search_call":
            action = payload.get("action") if isinstance(payload.get("action"), dict) else {}
            tool_name = str(action.get("type") or "web_search").strip() or "web_search"
            tool_event = tool_call_event(name=tool_name, args=action or {})
            tool_event.t = timestamp
            observation_payload = {
                "status": payload.get("status"),
            }
            results = payload.get("results")
            if results is not None:
                observation_payload["results"] = results
            tool_event.observation = observation_payload
            tool_event.extra = {"source": "web_search_call"}
            if pending_reasoning:
                _flush_reasoning(tool_event)
            events.append(tool_event)
            turn_event_indexes.append(len(events) - 1)
            continue

    _flush_reasoning()
    return _finalize_parse(
        instruction=instruction,
        parsed_events=events,
        metrics=metrics,
        extra=extra or {},
    )


def parse_codex_exec_output(stdout: str, *, instruction: str) -> ParseResult | None:
    raw_events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            raw_events.append(obj)
    if not raw_events:
        return None

    events: list[TranscriptEvent] = []
    metrics: dict[str, float] = {}
    pending_tool_calls: dict[str, TranscriptEvent] = {}
    turn_event_indexes: list[int] = []
    for raw_event in raw_events:
        event_type = raw_event.get("type")
        timestamp = safe_parse_timestamp(raw_event.get("timestamp")) or utcnow()
        if event_type == "turn.completed":
            usage = raw_event.get("usage") if isinstance(raw_event.get("usage"), dict) else {}
            _merge_metrics(metrics, usage)
            _attach_usage_to_last_turn_event(events, turn_event_indexes, usage)
            turn_event_indexes = []
            continue
        if event_type != "item.completed":
            continue
        item = raw_event.get("item") if isinstance(raw_event.get("item"), dict) else {}
        item_type = item.get("type")
        if item_type == "agent_message":
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            events.append(message_event(role="assistant", content=text, t=timestamp))
            turn_event_indexes.append(len(events) - 1)
            continue
        if item_type != "command_execution":
            continue
        call_id = str(item.get("id") or "").strip()
        args = {"cmd": item.get("command")} if item.get("command") is not None else {}
        tool_event = pending_tool_calls.get(call_id)
        if tool_event is None:
            tool_event = tool_call_event(name="exec_command", args=args)
            tool_event.t = timestamp
            tool_event.tool_call.id = call_id or None  # type: ignore[union-attr]
            pending_tool_calls[call_id] = tool_event
            events.append(tool_event)
            turn_event_indexes.append(len(events) - 1)
        observation_payload = {
            "stdout": item.get("aggregated_output"),
            "exit_code": item.get("exit_code"),
            "status": item.get("status"),
            "command": item.get("command"),
        }
        tool_event.observation = observation_payload
        tool_event.extra = {"source": "codex_exec_json_stdout"}

    return _finalize_parse(
        instruction=instruction,
        parsed_events=events,
        metrics=metrics,
        extra={"source": "codex-exec-output"},
    )


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

    return _parse_codex_session_events(
        raw_events,
        instruction=instruction,
        extra={"source": "codex-session", "session_file": session_file.name},
    )
