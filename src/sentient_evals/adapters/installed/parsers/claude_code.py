from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .common import ParseResult, message_event, tool_call_event, utcnow
from ....models import TranscriptEvent

logger = logging.getLogger(__name__)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _extract_text_reasoning_tool_uses(content: Any) -> tuple[str, str | None, list[dict[str, Any]]]:
    if isinstance(content, str):
        return content.strip(), None, []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_blocks: list[dict[str, Any]] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                text_parts.append(_stringify(block))
                continue
            bt = block.get("type")
            if bt == "tool_use":
                tool_blocks.append(block)
                continue
            if bt in {"thinking", "reasoning", "analysis"}:
                tv = block.get("text")
                reasoning_parts.append(tv.strip() if isinstance(tv, str) else _stringify(tv))
                continue
            if bt == "code" and isinstance(block.get("code"), str):
                text_parts.append(block["code"])
                continue
            tv = block.get("text")
            text_parts.append(tv if isinstance(tv, str) else _stringify(block))
    elif content is not None:
        text_parts.append(_stringify(content))
    text = "\n\n".join(p.strip() for p in text_parts if p and str(p).strip())
    reasoning = "\n\n".join(p.strip() for p in reasoning_parts if p and str(p).strip())
    return text, (reasoning or None), tool_blocks


def _build_metrics(usage: Any) -> dict[str, float] | None:
    if not isinstance(usage, dict):
        return None
    cached = int(usage.get("cache_read_input_tokens") or 0)
    prompt = int(usage.get("input_tokens") or 0) + cached
    completion = int(usage.get("output_tokens") or 0)
    extra: dict[str, Any] = {k: v for k, v in usage.items() if k not in {"input_tokens", "output_tokens"}}
    out: dict[str, float] = {}
    if prompt:
        out["prompt_tokens"] = float(prompt)
    if completion:
        out["completion_tokens"] = float(completion)
    if cached:
        out["cached_tokens"] = float(cached)
    if extra:
        out["extra"] = extra  # type: ignore[assignment]
    return out or None


def _format_tool_result(block: dict[str, Any], tool_use_result: dict[str, Any] | None) -> tuple[str | None, dict[str, Any] | None]:
    parts: list[str] = []
    content = block.get("content")
    if isinstance(content, str) and content.strip():
        parts.append(content.strip())
    elif isinstance(content, list):
        for item in content:
            s = _stringify(item).strip()
            if s:
                parts.append(s)
    elif content not in (None, ""):
        parts.append(_stringify(content))
    meta: dict[str, Any] | None = None
    if tool_use_result and isinstance(tool_use_result, dict):
        meta = {"tool_use_result": tool_use_result}
        stdout = tool_use_result.get("stdout")
        stderr = tool_use_result.get("stderr")
        exit_code = tool_use_result.get("exitCode") or tool_use_result.get("exit_code")
        chunks: list[str] = []
        if stdout:
            chunks.append(f"[stdout]\n{stdout}".rstrip())
        if stderr:
            chunks.append(f"[stderr]\n{stderr}".rstrip())
        if exit_code is not None:
            chunks.append(f"[exit_code] {exit_code}")
        if chunks:
            parts.append("\n".join(chunks))
    out = "\n\n".join(p for p in parts if p and str(p).strip())
    return (out or None), meta


def _find_session_file(agent_logs_dir: Path) -> Path | None:
    sessions_root = agent_logs_dir / "sessions"
    if not sessions_root.exists():
        return None
    project_root = sessions_root / "projects"
    candidates = list(project_root.glob("**/*.jsonl")) if project_root.exists() else []
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.stat().st_mtime, p.as_posix()))
    return candidates[-1]


def parse_claude_code_session(agent_logs_dir: Path, *, instruction: str) -> ParseResult | None:
    session_file = _find_session_file(agent_logs_dir)
    if session_file is None:
        return None
    try:
        events_raw = [json.loads(l) for l in session_file.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    except Exception:
        return None
    events_raw = [e for e in events_raw if isinstance(e, dict)]
    if not events_raw:
        return None

    normalized: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}
    default_model = None
    for e in events_raw:
        msg = e.get("message")
        if isinstance(msg, dict):
            mn = msg.get("model")
            if isinstance(mn, str) and mn:
                default_model = mn
                break

    for e in events_raw:
        msg = e.get("message")
        if not isinstance(msg, dict):
            continue
        et = e.get("type")
        ts = e.get("timestamp")
        if et == "assistant":
            text, reasoning, tool_blocks = _extract_text_reasoning_tool_uses(msg.get("content"))
            m = _build_metrics(msg.get("usage"))
            extra: dict[str, Any] = {"is_sidechain": bool(e.get("isSidechain", False))}
            for k in ("stop_reason", "stop_sequence", "requestId"):
                if msg.get(k) is not None:
                    extra[k] = msg.get(k)
            if e.get("id"):
                extra["id"] = e.get("id")
            if e.get("agent_id"):
                extra["agent_id"] = e.get("agent_id")
            if e.get("cwd"):
                extra["cwd"] = e.get("cwd")
            model = msg.get("model") or default_model
            if text or reasoning or not tool_blocks:
                normalized.append(
                    {
                        "kind": "message",
                        "timestamp": ts,
                        "role": msg.get("role", "assistant"),
                        "text": text or "",
                        "reasoning": reasoning if msg.get("role") == "assistant" else None,
                        "metrics": m,
                        "extra": extra or None,
                        "model_name": model,
                    }
                )
                m = None
            for idx, tb in enumerate(tool_blocks):
                call_id = tb.get("id") or tb.get("tool_use_id")
                if not call_id:
                    continue
                raw_args = tb.get("input")
                args = raw_args if isinstance(raw_args, dict) else {"input": raw_args}
                call_extra = dict(extra)
                if tb.get("is_error") is not None:
                    call_extra["tool_use_is_error"] = tb.get("is_error")
                pending[str(call_id)] = {
                    "kind": "tool_call",
                    "timestamp": ts,
                    "call_id": str(call_id),
                    "tool_name": tb.get("name") or "",
                    "arguments": args,
                    "raw_arguments": raw_args,
                    "reasoning": reasoning,
                    "status": tb.get("status"),
                    "message": None,
                    "extra": call_extra or None,
                    "metrics": m if idx == 0 and m is not None else None,
                    "model_name": model,
                }
                if idx == 0 and m is not None:
                    m = None
            continue

        if et == "user":
            content = msg.get("content")
            if isinstance(content, list):
                text_parts: list[str] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        call_id = block.get("tool_use_id")
                        out, meta = _format_tool_result(block, e.get("toolUseResult") if isinstance(e.get("toolUseResult"), dict) else None)
                        call_info = pending.pop(str(call_id), None) if call_id else None
                        if call_info is None:
                            call_info = {"kind": "tool_call", "timestamp": ts, "call_id": str(call_id or ""), "tool_name": block.get("name") or "", "arguments": {}, "raw_arguments": None, "reasoning": None, "status": None, "message": None, "extra": None, "metrics": None, "model_name": default_model}
                        extra_val = call_info.get("extra")
                        extra2: dict[str, Any] = extra_val if isinstance(extra_val, dict) else {}
                        if meta:
                            extra2["tool_result_metadata"] = meta
                        call_info["extra"] = extra2 or None
                        call_info["output"] = out
                        normalized.append(call_info)
                        continue
                    text_parts.append(_stringify(block))
                text_msg = "\n\n".join(p.strip() for p in text_parts if p and str(p).strip())
                if text_msg:
                    normalized.append({"kind": "message", "timestamp": ts, "role": "user", "text": text_msg})
                continue
            if isinstance(content, str) and content.strip():
                normalized.append({"kind": "message", "timestamp": ts, "role": "user", "text": content.strip()})
                continue

    for v in pending.values():
        normalized.append(v)

    out_events: list[TranscriptEvent] = [message_event(role="user", content=instruction, t=utcnow())]
    for ne in normalized:
        if ne.get("kind") == "message":
            ev = message_event(role=ne.get("role") or "assistant", content=ne.get("text") or "")
            ev.reasoning_content = ne.get("reasoning")
            ev.metrics = ne.get("metrics")
            ev.extra = ne.get("extra")
            out_events.append(ev)
        elif ne.get("kind") == "tool_call":
            tc = tool_call_event(name=ne.get("tool_name") or "tool", args=ne.get("arguments") or {}, observation=ne.get("output"))
            tc.tool_call.id = ne.get("call_id")  # type: ignore[union-attr]
            tc.reasoning_content = ne.get("reasoning")
            tc.metrics = ne.get("metrics")
            tc.extra = ne.get("extra")
            out_events.append(tc)

    prompt_total = 0
    completion_total = 0
    cached_total = 0
    for e in out_events:
        if not e.metrics:
            continue
        prompt_total += int(e.metrics.get("prompt_tokens") or 0)
        completion_total += int(e.metrics.get("completion_tokens") or 0)
        cached_total += int(e.metrics.get("cached_tokens") or 0)

    totals: dict[str, float] = {}
    if prompt_total:
        totals["prompt_tokens"] = float(prompt_total)
    if completion_total:
        totals["completion_tokens"] = float(completion_total)
    if cached_total:
        totals["cached_tokens"] = float(cached_total)
    return ParseResult(events=out_events, metrics=totals, extra={"source": "claude-code-session", "session_file": session_file.name})

