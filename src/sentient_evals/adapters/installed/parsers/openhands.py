from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .common import ParseResult, message_event, tool_call_event, utcnow
from ....models import TranscriptEvent

logger = logging.getLogger(__name__)


def _find_session_dir(agent_logs_dir: Path) -> Path | None:
    sessions = agent_logs_dir / "sessions"
    if not sessions.exists():
        return None
    candidates = [p for p in sessions.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _extract_tool_definitions(events_dir: Path) -> list[dict[str, Any]] | None:
    for p in sorted(events_dir.glob("*.json"), key=lambda x: int(x.stem) if x.stem.isdigit() else 0):
        try:
            ev = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(ev, dict) and isinstance(ev.get("args"), dict) and isinstance(ev["args"].get("tools"), list):
            tools = ev["args"]["tools"]
            if tools and all(isinstance(t, dict) for t in tools):
                return tools  # type: ignore[return-value]
    return None


def parse_openhands_session(agent_logs_dir: Path, *, instruction: str) -> ParseResult | None:
    session_dir = _find_session_dir(agent_logs_dir)
    if session_dir is None:
        return None
    events_dir = session_dir / "events"
    if not events_dir.exists():
        return None

    files = sorted(events_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    raw: list[dict[str, Any]] = []
    for f in files:
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(obj, dict):
            raw.append(obj)
    if not raw:
        return None

    tool_defs = _extract_tool_definitions(events_dir)

    out: list[TranscriptEvent] = [message_event(role="user", content=instruction, t=utcnow())]
    prev_acc_prompt = 0
    prev_acc_completion = 0
    prev_acc_cached = 0
    prev_acc_cost = 0.0

    last_tool_call_id: str | None = None
    pending_tool: TranscriptEvent | None = None

    for ev in raw:
        source = ev.get("source") or ""
        message = ev.get("message") or ""
        if ev.get("action") == "system" or source == "environment":
            source = "system"
        if source == "user" and isinstance(message, str):
            system_prefixes = ("Retrieving content for:", "Added workspace context", "Loading workspace", "Initializing")
            if any(message.startswith(p) for p in system_prefixes):
                source = "system"

        role = "assistant"
        if source == "user":
            role = "user"
        elif source == "system":
            role = "system"

        metrics: dict[str, float] | None = None
        llm_metrics = ev.get("llm_metrics") if isinstance(ev.get("llm_metrics"), dict) else None
        if llm_metrics and source == "agent":
            usage = llm_metrics.get("accumulated_token_usage") if isinstance(llm_metrics.get("accumulated_token_usage"), dict) else {}
            acc_prompt = int(usage.get("prompt_tokens") or 0)
            acc_completion = int(usage.get("completion_tokens") or 0)
            acc_cached = int(usage.get("cache_read_tokens") or 0)
            acc_cost = float(llm_metrics.get("accumulated_cost") or 0.0)
            d_prompt = max(0, acc_prompt - prev_acc_prompt)
            d_completion = max(0, acc_completion - prev_acc_completion)
            d_cached = max(0, acc_cached - prev_acc_cached)
            d_cost = max(0.0, acc_cost - prev_acc_cost)
            prev_acc_prompt, prev_acc_completion, prev_acc_cached, prev_acc_cost = acc_prompt, acc_completion, acc_cached, acc_cost
            metrics = {}
            if d_prompt:
                metrics["prompt_tokens"] = float(d_prompt)
            if d_completion:
                metrics["completion_tokens"] = float(d_completion)
            if d_cached:
                metrics["cached_tokens"] = float(d_cached)
            if d_cost:
                metrics["cost_usd"] = float(d_cost)

        tc_meta = ev.get("tool_call_metadata") if isinstance(ev.get("tool_call_metadata"), dict) else None
        obs_present = "observation" in ev and ev.get("cause") is not None and source == "agent"

        if tc_meta and source == "agent":
            tool_call_id = str(tc_meta.get("tool_call_id") or "")
            fn = str(tc_meta.get("function_name") or "")
            args: dict[str, Any] = {}
            model_resp = tc_meta.get("model_response")
            if isinstance(model_resp, dict):
                try:
                    tc_list = (((model_resp.get("choices") or [])[0] or {}).get("message") or {}).get("tool_calls") or []
                    if tc_list:
                        func = (tc_list[0] or {}).get("function") or {}
                        raw_args = func.get("arguments")
                        if isinstance(raw_args, str):
                            args = json.loads(raw_args)
                        elif isinstance(raw_args, dict):
                            args = raw_args
                except Exception:
                    args = {}
            tool_ev = tool_call_event(name=fn or "tool", args=args, observation=None)
            tool_ev.tool_call.id = tool_call_id or None  # type: ignore[union-attr]
            tool_ev.metrics = metrics
            
            if pending_tool is not None:
                out.append(pending_tool)
            pending_tool = tool_ev
            last_tool_call_id = tool_call_id or None
            continue

        if obs_present:
            obs_text = ev.get("content") or ""
            if pending_tool and last_tool_call_id and pending_tool.tool_call and pending_tool.tool_call.id == last_tool_call_id:
                pending_tool.observation = obs_text
                out.append(pending_tool)
                pending_tool = None
                continue
            obs_ev = tool_call_event(name="tool", args={}, observation=obs_text)
            if last_tool_call_id:
                obs_ev.tool_call.id = last_tool_call_id  # type: ignore[union-attr]
            obs_ev.metrics = metrics
            out.append(obs_ev)
            continue

        if isinstance(message, str) and message.strip():
            msg_ev = message_event(role=role, content=message.strip())
            msg_ev.metrics = metrics
            out.append(msg_ev)

    if pending_tool is not None:
        out.append(pending_tool)

    return ParseResult(events=out, metrics={}, tool_definitions=tool_defs, extra={"source": "openhands-events", "session": session_dir.name})

