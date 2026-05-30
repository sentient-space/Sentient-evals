from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .common import ParseResult, message_event, safe_parse_timestamp, tool_call_event, try_parse_json_lines, utcnow
from ....models import TranscriptEvent


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0)
    except Exception:
        return 0.0


def _jsonish(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str) and item.strip():
            parts.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts).strip()


def _text_from_message(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    text = _text_from_content(message.get("content"))
    if text:
        return text
    return _text_from_content(message.get("parts"))


def _normalized_role(value: Any) -> str:
    role = str(value or "").lower()
    if role in {"assistant", "agent", "model", "gemini"}:
        return "assistant"
    if role == "user":
        return "user"
    return "assistant"


def _usage_metrics(usage: Any) -> dict[str, float]:
    if not isinstance(usage, dict):
        return {}
    input_tokens = _as_int(
        usage.get("inputTokens")
        or usage.get("input_tokens")
        or usage.get("promptTokenCount")
        or usage.get("prompt_tokens")
    )
    output_tokens = _as_int(
        usage.get("outputTokens")
        or usage.get("output_tokens")
        or usage.get("candidatesTokenCount")
        or usage.get("completion_tokens")
    )
    cache_read = _as_int(
        usage.get("cacheReadTokens")
        or usage.get("cachedContentTokenCount")
        or usage.get("cached_tokens")
    )
    cache_write = _as_int(usage.get("cacheWriteTokens"))
    thoughts = _as_int(usage.get("thoughtsTokenCount") or usage.get("thoughts_tokens"))

    metrics: dict[str, float] = {}
    prompt_tokens = input_tokens + cache_read + cache_write
    completion_tokens = output_tokens + thoughts
    if prompt_tokens:
        metrics["prompt_tokens"] = float(prompt_tokens)
    if completion_tokens:
        metrics["completion_tokens"] = float(completion_tokens)
    if cache_read:
        metrics["cached_tokens"] = float(cache_read)
    return metrics


def _merge_metrics(target: dict[str, float], update: dict[str, float]) -> None:
    for key, value in update.items():
        target[key] = target.get(key, 0.0) + float(value)


def _tool_blocks_from_message(message: Any) -> list[dict[str, Any]]:
    if not isinstance(message, dict):
        return []
    blocks: list[dict[str, Any]] = []
    for key in ("content", "parts"):
        raw_blocks = message.get(key)
        if not isinstance(raw_blocks, list):
            continue
        for block in raw_blocks:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "").lower()
            if block_type in {"tool_use", "tool-call", "tool_call"}:
                blocks.append(block)
            function_call = block.get("functionCall")
            if isinstance(function_call, dict):
                blocks.append({"type": "functionCall", **function_call})
    return blocks


def _emit_tool_block(block: dict[str, Any], *, observation: Any = None) -> TranscriptEvent | None:
    name = block.get("name") or block.get("tool") or block.get("function_name")
    if not name:
        return None
    args = block.get("input") or block.get("args") or block.get("arguments") or {}
    if not isinstance(args, dict):
        args = {"value": args}
    ev = tool_call_event(name=str(name), args=args, observation=observation)
    call_id = block.get("id") or block.get("call_id") or block.get("toolCallId")
    if call_id:
        ev.tool_call.id = str(call_id)  # type: ignore[union-attr]
    return ev


def parse_gemini_stream(agent_logs_dir: Path, *, instruction: str) -> ParseResult | None:
    output_path = agent_logs_dir / "gemini-cli.txt"
    raw_events = try_parse_json_lines(output_path)
    if not raw_events:
        return None

    events: list[TranscriptEvent] = [message_event(role="user", content=instruction, t=utcnow())]
    metrics: dict[str, float] = {}
    session_id: str | None = None
    model_name: str | None = None
    saw_meaningful_event = False
    pending_tools: dict[str, TranscriptEvent] = {}

    for raw in raw_events:
        event_type = str(raw.get("type") or raw.get("event") or "").lower()
        session_id = str(raw.get("session_id") or raw.get("sessionId") or session_id or "").strip() or session_id
        model_name = str(raw.get("model") or model_name or "").strip() or model_name

        if event_type in {"message", "assistant"}:
            message = raw.get("message") if isinstance(raw.get("message"), dict) else raw
            role = str(message.get("role") or raw.get("role") or "assistant").lower()
            text = _text_from_message(message)
            if text:
                events.append(message_event(role=_normalized_role(role), content=text))
                saw_meaningful_event = True
            for block in _tool_blocks_from_message(message):
                ev = _emit_tool_block(block)
                if ev is None:
                    continue
                ev.extra = {"source": "gemini-stream"}
                call_id = ev.tool_call.id if ev.tool_call else None
                if call_id:
                    pending_tools[call_id] = ev
                events.append(ev)
                saw_meaningful_event = True
            _merge_metrics(metrics, _usage_metrics(raw.get("usage") or message.get("usage")))
            continue

        if event_type in {"tool_use", "tool_call"}:
            ev = _emit_tool_block(raw)
            if ev is None:
                continue
            ev.extra = {"source": "gemini-stream"}
            call_id = ev.tool_call.id if ev.tool_call else None
            if call_id:
                pending_tools[call_id] = ev
            events.append(ev)
            saw_meaningful_event = True
            continue

        if event_type in {"tool_result", "tool_response"}:
            call_id = str(raw.get("id") or raw.get("call_id") or raw.get("tool_use_id") or "").strip()
            observation = raw.get("result") or raw.get("output") or raw.get("content")
            if call_id and call_id in pending_tools:
                pending_tools[call_id].observation = observation
            elif observation is not None:
                events.append(tool_call_event(name="tool", args={}, observation=observation))
            saw_meaningful_event = True
            continue

        if event_type == "result":
            _merge_metrics(metrics, _usage_metrics(raw.get("usage")))
            result = raw.get("result") or raw.get("response")
            if isinstance(result, str) and result.strip():
                events.append(message_event(role="assistant", content=result.strip()))
                saw_meaningful_event = True

    if not saw_meaningful_event:
        return None
    extra: dict[str, Any] = {"source": "gemini-stream", "output_file": output_path.name}
    if session_id:
        extra["session_id"] = session_id
    if model_name:
        extra["model"] = model_name
    return ParseResult(events=events, metrics=metrics, extra=extra)


def parse_qwen_code_stream(agent_logs_dir: Path, *, instruction: str) -> ParseResult | None:
    output_path = agent_logs_dir / "qwen-code.txt"
    raw_events = try_parse_json_lines(output_path)
    if not raw_events:
        return None

    events: list[TranscriptEvent] = [message_event(role="user", content=instruction, t=utcnow())]
    metrics: dict[str, float] = {}
    session_id: str | None = None
    model_name: str | None = None
    saw_meaningful_event = False
    pending_tools: dict[str, TranscriptEvent] = {}

    for raw in raw_events:
        event_type = str(raw.get("type") or "").lower()
        session_id = str(raw.get("session_id") or raw.get("sessionId") or session_id or "").strip() or session_id
        model_name = str(raw.get("model") or model_name or "").strip() or model_name
        message = raw.get("message")

        if event_type == "system":
            continue

        if event_type in {"assistant", "user"} and isinstance(message, dict):
            role = str(message.get("role") or event_type).lower()
            text = _text_from_message(message)
            if text and (role != "user" or text.strip() != instruction.strip()):
                events.append(message_event(role=_normalized_role(role), content=text))
                saw_meaningful_event = True
            for block in _tool_blocks_from_message(message):
                ev = _emit_tool_block(block)
                if ev is None:
                    continue
                ev.extra = {"source": "qwen-code-stream"}
                call_id = ev.tool_call.id if ev.tool_call else None
                if call_id:
                    pending_tools[call_id] = ev
                events.append(ev)
                saw_meaningful_event = True
            _merge_metrics(metrics, _usage_metrics(raw.get("usage") or message.get("usage") or raw.get("usageMetadata")))
            continue

        if event_type in {"tool_result", "tool_response"} and isinstance(message, dict):
            for block in message.get("parts") or message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                response = block.get("functionResponse") or {}
                if not isinstance(response, dict):
                    continue
                call_id = str(response.get("id") or "").strip()
                observation = (response.get("response") or {}).get("output") if isinstance(response.get("response"), dict) else response.get("response")
                if call_id and call_id in pending_tools:
                    pending_tools[call_id].observation = observation
                elif observation is not None:
                    events.append(tool_call_event(name="tool", args={}, observation=observation))
                saw_meaningful_event = True
            continue

        if event_type == "result":
            _merge_metrics(metrics, _usage_metrics(raw.get("usage")))
            result = raw.get("result") or raw.get("response")
            if isinstance(result, str) and result.strip():
                events.append(message_event(role="assistant", content=result.strip()))
                saw_meaningful_event = True

    if not saw_meaningful_event:
        return None
    extra: dict[str, Any] = {"source": "qwen-code-stream", "output_file": output_path.name}
    if session_id:
        extra["session_id"] = session_id
    if model_name:
        extra["model"] = model_name
    return ParseResult(events=events, metrics=metrics, extra=extra)


def parse_cline_json_stream(agent_logs_dir: Path, *, instruction: str) -> ParseResult | None:
    output_path = agent_logs_dir / "cline.txt"
    raw_events = try_parse_json_lines(output_path)
    if not raw_events:
        return None

    events: list[TranscriptEvent] = [message_event(role="user", content=instruction, t=utcnow())]
    saw_meaningful_event = False
    tool_name_pattern = re.compile(r"\b(?:tool|command|running|ran|execute|executing)\s+([A-Za-z0-9_.:-]+)", re.IGNORECASE)

    for raw in raw_events:
        event_type = str(raw.get("type") or "").lower()
        subtype = str(raw.get("say") or raw.get("ask") or "").lower()
        text = str(raw.get("text") or "").strip()
        timestamp = safe_parse_timestamp(raw.get("ts")) or utcnow()
        if not text:
            continue
        if subtype == "tool":
            match = tool_name_pattern.search(text)
            name = match.group(1) if match else "cline_tool"
            ev = tool_call_event(name=name, args={"text": text}, observation=None)
            ev.t = timestamp
            ev.extra = {"source": "cline-json", "event_type": event_type, "subtype": subtype}
            events.append(ev)
        else:
            ev = message_event(role="assistant", content=text, t=timestamp)
            ev.extra = {"source": "cline-json", "event_type": event_type, "subtype": subtype or None}
            events.append(ev)
        saw_meaningful_event = True

    if not saw_meaningful_event:
        return None
    return ParseResult(
        events=events,
        metrics={},
        extra={"source": "cline-json", "output_file": output_path.name},
    )


def parse_opencode_json_stream(agent_logs_dir: Path, *, instruction: str, model_name: str | None = None) -> ParseResult | None:
    output_path = agent_logs_dir / "opencode.txt"
    raw_events = try_parse_json_lines(output_path)
    if not raw_events:
        return None

    events: list[TranscriptEvent] = [message_event(role="user", content=instruction, t=utcnow())]
    metrics: dict[str, float] = {}
    saw_meaningful_event = False
    session_id: str | None = None

    for raw in raw_events:
        event_type = str(raw.get("type") or "").lower()
        part = raw.get("part") if isinstance(raw.get("part"), dict) else {}
        session_id = str(raw.get("sessionID") or raw.get("session_id") or session_id or "").strip() or session_id
        timestamp = safe_parse_timestamp(raw.get("timestamp")) or utcnow()

        if event_type == "text" or part.get("type") == "text":
            text = str(part.get("text") or raw.get("text") or "").strip()
            if text:
                ev = message_event(role="assistant", content=text, t=timestamp)
                ev.extra = {"source": "opencode-json", "message_id": part.get("messageID")}
                events.append(ev)
                saw_meaningful_event = True
            continue

        if event_type == "tool_use" or part.get("type") == "tool":
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            raw_input = state.get("input")
            args = raw_input if isinstance(raw_input, dict) else {"value": raw_input} if raw_input is not None else {}
            ev = tool_call_event(
                name=str(part.get("tool") or raw.get("tool") or "tool"),
                args=args,
                observation=state.get("output"),
            )
            ev.t = timestamp
            call_id = part.get("callID") or part.get("callId") or raw.get("callID") or raw.get("call_id")
            if call_id:
                ev.tool_call.id = str(call_id)  # type: ignore[union-attr]
            ev.extra = {"source": "opencode-json", "message_id": part.get("messageID"), "status": state.get("status")}
            events.append(ev)
            saw_meaningful_event = True
            continue

        if event_type == "step_finish" or part.get("type") == "step-finish":
            tokens = part.get("tokens") if isinstance(part.get("tokens"), dict) else {}
            cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
            prompt_tokens = _as_int(tokens.get("input")) + _as_int(cache.get("read"))
            completion_tokens = _as_int(tokens.get("output")) + _as_int(tokens.get("reasoning"))
            if prompt_tokens:
                metrics["prompt_tokens"] = metrics.get("prompt_tokens", 0.0) + float(prompt_tokens)
            if completion_tokens:
                metrics["completion_tokens"] = metrics.get("completion_tokens", 0.0) + float(completion_tokens)
            if cache.get("read"):
                metrics["cached_tokens"] = metrics.get("cached_tokens", 0.0) + float(_as_int(cache.get("read")))
            cost = _as_float(part.get("cost"))
            if cost:
                metrics["cost_usd"] = metrics.get("cost_usd", 0.0) + cost
            continue

    if not saw_meaningful_event:
        return None
    extra: dict[str, Any] = {"source": "opencode-json", "output_file": output_path.name}
    if session_id:
        extra["session_id"] = session_id
    if model_name:
        extra["model"] = model_name
    return ParseResult(events=events, metrics=metrics, extra=extra)


def parse_goose_stream(agent_logs_dir: Path, *, instruction: str, model_name: str | None = None) -> ParseResult | None:
    output_path = agent_logs_dir / "goose.txt"
    raw_events = try_parse_json_lines(output_path)
    if not raw_events:
        return None

    events: list[TranscriptEvent] = [message_event(role="user", content=instruction, t=utcnow())]
    metrics: dict[str, float] = {}
    messages: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    saw_meaningful_event = False

    for raw in raw_events:
        event_type = str(raw.get("type") or "").lower()
        if event_type == "complete":
            total = _as_int(raw.get("total_tokens"))
            if total:
                metrics["total_tokens"] = float(total)
            continue
        if event_type == "error":
            events.append(message_event(role="assistant", content=f"[error] {raw.get('error', 'unknown error')}"))
            saw_meaningful_event = True
            continue
        if event_type != "message" or not isinstance(raw.get("message"), dict):
            continue
        msg = raw["message"]
        msg_id = str(msg.get("id") or len(messages))
        role = str(msg.get("role") or "").lower()
        entry = messages.setdefault(msg_id, {"role": role, "text": [], "tools": [], "observations": []})
        for item in msg.get("content") or []:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").lower()
            if item_type in {"text", "thinking"} and item.get("text"):
                entry["text"].append(str(item["text"]))
            elif item_type == "toolrequest":
                call = ((item.get("toolCall") or {}).get("value") or {}) if isinstance(item.get("toolCall"), dict) else {}
                ev = tool_call_event(
                    name=str(call.get("name") or "tool"),
                    args=call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
                    observation=None,
                )
                ev.tool_call.id = str(item.get("id")) if item.get("id") else None  # type: ignore[union-attr]
                entry["tools"].append(ev)
            elif item_type == "toolresponse":
                result = ((item.get("toolResult") or {}).get("value") or {}) if isinstance(item.get("toolResult"), dict) else {}
                content = result.get("content")
                obs = _jsonish(content if content is not None else result)
                entry["observations"].append((str(item.get("id") or ""), obs))

    for entry in messages.values():
        role = entry["role"]
        text = "".join(entry["text"]).strip()
        if role == "user" and entry["observations"] and events:
            for call_id, obs in entry["observations"]:
                for prior in reversed(events):
                    if prior.tool_call and (not call_id or prior.tool_call.id == call_id):
                        prior.observation = obs
                        break
            continue
        if role == "assistant":
            if text:
                events.append(message_event(role="assistant", content=text))
                saw_meaningful_event = True
            for ev in entry["tools"]:
                ev.extra = {"source": "goose-stream"}
                events.append(ev)
                saw_meaningful_event = True

    if not saw_meaningful_event:
        return None
    extra: dict[str, Any] = {"source": "goose-stream", "output_file": output_path.name}
    if model_name:
        extra["model"] = model_name
    return ParseResult(events=events, metrics=metrics, extra=extra)
