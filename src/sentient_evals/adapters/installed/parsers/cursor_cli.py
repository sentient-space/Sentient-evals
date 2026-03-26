from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import ParseResult, message_event, safe_parse_timestamp, tool_call_event, try_parse_json_lines, utcnow
from ....models import TranscriptEvent


def _extract_message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            text = item.strip()
            if text:
                parts.append(text)
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts)


def _consume_reasoning(pending: list[str]) -> str | None:
    reasoning = "\n\n".join(part.strip() for part in pending if isinstance(part, str) and part.strip()).strip()
    pending.clear()
    return reasoning or None


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _build_usage_metrics(usage: Any) -> dict[str, float]:
    if not isinstance(usage, dict):
        return {}
    input_tokens = _as_int(usage.get("inputTokens", usage.get("input_tokens")))
    output_tokens = _as_int(usage.get("outputTokens", usage.get("output_tokens")))
    cache_read_tokens = _as_int(usage.get("cacheReadTokens", usage.get("cache_read_tokens")))
    cache_write_tokens = _as_int(usage.get("cacheWriteTokens", usage.get("cache_write_tokens")))

    metrics: dict[str, float] = {}
    prompt_tokens = input_tokens + cache_read_tokens + cache_write_tokens
    if prompt_tokens:
        metrics["prompt_tokens"] = float(prompt_tokens)
    if output_tokens:
        metrics["completion_tokens"] = float(output_tokens)
    if cache_read_tokens:
        metrics["cached_tokens"] = float(cache_read_tokens)
    return metrics


def _normalize_tool_result_content(result: Any) -> str | None:
    if result is None:
        return None
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except Exception:
        return str(result)


def _parse_tool_arguments(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if raw_args is None:
        return {}
    return {"value": raw_args}


def parse_cursor_cli_stream(agent_logs_dir: Path, *, instruction: str) -> ParseResult | None:
    output_path = agent_logs_dir / "cursor-cli.txt"
    raw_events = try_parse_json_lines(output_path)
    if not raw_events:
        return None

    events: list[TranscriptEvent] = [message_event(role="user", content=instruction, t=utcnow())]
    metrics: dict[str, float] = {}
    pending_thinking: list[str] = []
    started_calls: dict[str, Any] = {}

    session_id: str | None = None
    model_name: str | None = None
    permission_mode: str | None = None
    api_key_source: str | None = None
    duration_ms = 0
    duration_api_ms = 0
    request_id: str | None = None
    result_subtypes: list[str] = []
    saw_meaningful_event = False

    instruction_text = instruction.strip()

    for raw_event in raw_events:
        event_type = raw_event.get("type")

        if event_type == "system":
            session_id = str(raw_event.get("session_id") or raw_event.get("sessionId") or session_id or "").strip() or session_id
            model_name = str(raw_event.get("model") or model_name or "").strip() or model_name
            permission_mode = str(raw_event.get("permissionMode") or permission_mode or "").strip() or permission_mode
            api_key_source = str(raw_event.get("apiKeySource") or api_key_source or "").strip() or api_key_source
            continue

        if event_type == "user":
            session_id = str(raw_event.get("session_id") or raw_event.get("sessionId") or session_id or "").strip() or session_id
            text = _extract_message_text(raw_event.get("message"))
            if text and text != instruction_text:
                ev = message_event(role="user", content=text, t=utcnow())
                ev.extra = {"source": "cursor-cli-stream"}
                events.append(ev)
                saw_meaningful_event = True
            continue

        if event_type == "thinking":
            session_id = str(raw_event.get("session_id") or raw_event.get("sessionId") or session_id or "").strip() or session_id
            text = raw_event.get("text")
            if isinstance(text, str) and text.strip():
                pending_thinking.append(text.strip())
            continue

        if event_type == "assistant":
            session_id = str(raw_event.get("session_id") or raw_event.get("sessionId") or session_id or "").strip() or session_id
            timestamp = safe_parse_timestamp(raw_event.get("timestamp_ms") or raw_event.get("timestampMs")) or utcnow()
            text = _extract_message_text(raw_event.get("message"))
            reasoning = _consume_reasoning(pending_thinking)
            if not text and not reasoning:
                continue
            ev = message_event(role="assistant", content=text, t=timestamp)
            ev.reasoning_content = reasoning
            extra: dict[str, Any] = {"source": "cursor-cli-stream"}
            model_call_id = raw_event.get("model_call_id") or raw_event.get("modelCallId")
            if model_call_id is not None:
                extra["model_call_id"] = model_call_id
            if session_id:
                extra["session_id"] = session_id
            ev.extra = extra
            events.append(ev)
            saw_meaningful_event = True
            continue

        if event_type == "tool_call":
            session_id = str(raw_event.get("session_id") or raw_event.get("sessionId") or session_id or "").strip() or session_id
            subtype = str(raw_event.get("subtype") or "").strip().lower()
            call_id = str(raw_event.get("call_id") or raw_event.get("callId") or "").strip()
            timestamp = safe_parse_timestamp(raw_event.get("timestamp_ms") or raw_event.get("timestampMs")) or utcnow()
            if subtype == "started":
                if call_id:
                    started_calls[call_id] = timestamp
                continue
            if subtype != "completed":
                continue
            tool_payload = raw_event.get("tool_call")
            if not isinstance(tool_payload, dict):
                continue
            reasoning = _consume_reasoning(pending_thinking)
            model_call_id = raw_event.get("model_call_id") or raw_event.get("modelCallId")
            for tool_name, tool_call in tool_payload.items():
                if not isinstance(tool_call, dict):
                    continue
                ev = tool_call_event(
                    name=str(tool_name or "tool"),
                    args=_parse_tool_arguments(tool_call.get("args")),
                    observation=_normalize_tool_result_content(tool_call.get("result")),
                )
                ev.t = timestamp
                ev.reasoning_content = reasoning
                ev.tool_call.id = call_id or None  # type: ignore[union-attr]
                ev.tool_call.started_at = started_calls.get(call_id) if call_id else None  # type: ignore[union-attr]
                ev.tool_call.ended_at = timestamp  # type: ignore[union-attr]
                extra: dict[str, Any] = {"source": "cursor-cli-stream", "subtype": subtype}
                if session_id:
                    extra["session_id"] = session_id
                if model_call_id is not None:
                    extra["model_call_id"] = model_call_id
                ev.extra = extra
                events.append(ev)
                saw_meaningful_event = True
                reasoning = None
            continue

        if event_type == "result":
            session_id = str(raw_event.get("session_id") or raw_event.get("sessionId") or session_id or "").strip() or session_id
            usage_metrics = _build_usage_metrics(raw_event.get("usage"))
            for key, value in usage_metrics.items():
                metrics[key] = metrics.get(key, 0.0) + value
            duration_ms += _as_int(raw_event.get("duration_ms") or raw_event.get("durationMs"))
            duration_api_ms += _as_int(raw_event.get("duration_api_ms") or raw_event.get("durationApiMs"))
            req_id = raw_event.get("request_id") or raw_event.get("requestId")
            if isinstance(req_id, str) and req_id.strip():
                request_id = req_id.strip()
            subtype = raw_event.get("subtype")
            if isinstance(subtype, str) and subtype.strip():
                result_subtypes.append(subtype.strip())
            continue

    if not saw_meaningful_event:
        return None

    extra: dict[str, Any] = {"source": "cursor-cli-stream", "output_file": output_path.name}
    if session_id:
        extra["session_id"] = session_id
    if model_name:
        extra["model"] = model_name
    if permission_mode:
        extra["permission_mode"] = permission_mode
    if api_key_source:
        extra["api_key_source"] = api_key_source
    if duration_ms:
        extra["duration_ms"] = duration_ms
    if duration_api_ms:
        extra["duration_api_ms"] = duration_api_ms
    if request_id:
        extra["request_id"] = request_id
    if result_subtypes:
        extra["result_subtypes"] = result_subtypes

    return ParseResult(events=events, metrics=metrics, extra=extra)
