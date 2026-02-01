from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .common import ParseResult, message_event, tool_call_event, safe_parse_timestamp, utcnow
from ....models import TranscriptEvent

logger = logging.getLogger(__name__)


def _extract_usage(message: dict[str, Any]) -> tuple[int, int, int]:
    usage = ((message.get("extra") or {}).get("response") or {}).get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    cached_tokens = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
    return prompt_tokens, completion_tokens, cached_tokens


def parse_mini_swe_agent_trajectory(
    path: Path, *, instruction: str, model_name: str | None
) -> ParseResult | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("failed to parse mini-swe-agent trajectory: %s", exc)
        return None

    events: list[TranscriptEvent] = [message_event(role="user", content=instruction, t=utcnow())]
    metrics: dict[str, float] = {}

    # Aggregate usage where available.
    total_prompt = 0
    total_completion = 0
    total_cached = 0
    for msg in data.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        p, c, k = _extract_usage(msg)
        total_prompt += p
        total_completion += c
        total_cached += k
    if total_prompt:
        metrics["prompt_tokens"] = float(total_prompt)
    if total_completion:
        metrics["completion_tokens"] = float(total_completion)
    if total_cached:
        metrics["cached_tokens"] = float(total_cached)

    # Cost (Harbor uses instance_cost)
    total_cost = ((data.get("info") or {}).get("model_stats") or {}).get("instance_cost") or 0
    if total_cost:
        metrics["cost_usd"] = float(total_cost)

    # Convert messages into a message timeline where possible.
    for msg in data.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or msg.get("source") or "assistant"
        # Normalize a few likely values.
        if role in {"assistant", "agent"}:
            r = "assistant"
        elif role == "user":
            r = "user"
        else:
            r = "assistant"
        content = msg.get("content") or msg.get("message") or ""
        events.append(message_event(role=r, content=content, t=utcnow()))

        # Some formats may include tool calls per message; keep best-effort.
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                call_id = tc.get("id")
                name = tc.get("name") or tc.get("function_name") or "tool"
                args = tc.get("arguments") or tc.get("args") or {}
                ev = tool_call_event(
                    name=str(name), args=args if isinstance(args, dict) else {"value": args}
                )
                ev.tool_call.id = str(call_id) if call_id else None  # type: ignore[union-attr]
                events.append(ev)

    return ParseResult(events=events, metrics=metrics, extra={"source": "mini-swe-agent-trajectory"})

