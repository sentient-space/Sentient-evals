from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .common import ParseResult, message_event, tool_call_event, utcnow
from ....models import TranscriptEvent

logger = logging.getLogger(__name__)


def parse_gemini_trajectory(agent_logs_dir: Path, *, instruction: str) -> ParseResult | None:
    traj_path = agent_logs_dir / "gemini-cli.trajectory.json"
    if not traj_path.exists():
        return None
    try:
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    messages = traj.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return None

    out: list[TranscriptEvent] = [message_event(role="user", content=instruction, t=utcnow())]
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}

    for m in messages:
        if not isinstance(m, dict):
            continue
        mt = m.get("type")
        if mt == "user":
            out.append(message_event(role="user", content=m.get("content") or ""))
            continue
        if mt != "gemini":
            continue
        content = m.get("content") or ""
        thoughts = m.get("thoughts") or []
        reasoning: str | None = None
        if isinstance(thoughts, list) and thoughts:
            parts: list[str] = []
            for t in thoughts:
                if not isinstance(t, dict):
                    continue
                subj = t.get("subject") or ""
                desc = t.get("description") or ""
                if subj and desc:
                    parts.append(f"{subj}: {desc}")
                elif desc:
                    parts.append(str(desc))
            reasoning = "\n".join(parts) if parts else None

        tokens = m.get("tokens") or {}
        metrics: dict[str, float] | None = None
        if isinstance(tokens, dict) and tokens:
            input_tokens = int(tokens.get("input") or 0)
            output_tokens = int(tokens.get("output") or 0)
            cached_tokens = int(tokens.get("cached") or 0)
            thoughts_tokens = int(tokens.get("thoughts") or 0)
            tool_tokens = int(tokens.get("tool") or 0)
            completion_tokens = output_tokens + thoughts_tokens + tool_tokens
            totals["prompt_tokens"] += input_tokens
            totals["completion_tokens"] += completion_tokens
            totals["cached_tokens"] += cached_tokens
            metrics = {
                "prompt_tokens": float(input_tokens),
                "completion_tokens": float(completion_tokens),
                "cached_tokens": float(cached_tokens),
                "extra": {"thoughts_tokens": thoughts_tokens, "tool_tokens": tool_tokens},
            }

        msg_ev = message_event(role="assistant", content=content)
        msg_ev.reasoning_content = reasoning
        msg_ev.metrics = metrics
        out.append(msg_ev)

        tool_calls = m.get("toolCalls") or []
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id") or ""
                name = tc.get("name") or "tool"
                args = tc.get("args") if isinstance(tc.get("args"), dict) else {}
                result = tc.get("result") or []
                obs: str | None = None
                if isinstance(result, list):
                    for r in result:
                        if isinstance(r, dict):
                            fr = r.get("functionResponse") or {}
                            resp = fr.get("response") if isinstance(fr, dict) else {}
                            if isinstance(resp, dict) and resp.get("output"):
                                obs = resp.get("output")
                                break
                tcev = tool_call_event(name=str(name), args=args, observation=obs)
                tcev.tool_call.id = str(tc_id) if tc_id else None  # type: ignore[union-attr]
                out.append(tcev)

    return ParseResult(
        events=out,
        metrics={k: float(v) for k, v in totals.items() if v},
        extra={"source": "gemini-trajectory", "trajectory_file": traj_path.name},
    )

