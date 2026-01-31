from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .common import ParseResult, message_event, tool_call_event, utcnow
from ....models import TranscriptEvent

logger = logging.getLogger(__name__)


def parse_swe_agent_traj(path: Path, *, instruction: str, model_name: str | None) -> ParseResult | None:
    """
    Best-effort parser for SWE-agent `.traj` JSON (copied/derived from Harbor conversion notes).
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("failed to parse swe-agent traj: %s", exc)
        return None

    events: list[TranscriptEvent] = [message_event(role="user", content=instruction, t=utcnow())]
    metrics: dict[str, float] = {}

    info = data.get("info") or {}
    # Common token/cost fields used in Harbor conversion.
    input_tokens = info.get("input_tokens") or 0
    output_tokens = info.get("output_tokens") or 0
    total_cost = info.get("total_cost") or info.get("cost") or 0.0
    if input_tokens:
        metrics["prompt_tokens"] = float(input_tokens)
    if output_tokens:
        metrics["completion_tokens"] = float(output_tokens)
    if total_cost:
        metrics["cost_usd"] = float(total_cost)

    traj = data.get("trajectory") or []
    if not isinstance(traj, list):
        traj = []

    if traj and isinstance(traj[0], dict) and isinstance(traj[0].get("query"), list):
        for msg in traj[0].get("query") or []:
            if isinstance(msg, dict) and msg.get("role") == "system":
                events.insert(0, message_event(role="system", content=msg.get("content") or "", t=utcnow()))
                break

    for step in traj:
        if not isinstance(step, dict):
            continue

        # Agent response (often raw LLM output).
        response = step.get("response")
        thought = step.get("thought")
        if response:
            ev = message_event(role="assistant", content=str(response), t=utcnow())
            if isinstance(thought, str) and thought.strip():
                ev.reasoning_content = thought
            events.append(ev)

        action = step.get("action")
        observation = step.get("observation")
        if action:
            # Treat each action as a tool call; SWE-agent actions are strings.
            tc = tool_call_event(
                name="swe_agent_action",
                args={"raw_action": str(action)},
                observation=str(observation) if observation is not None else None,
            )
            tc.tool_call.id = f"call_{len(events)}_1"  # type: ignore[union-attr]
            events.append(tc)

    extra = {
        "source": "swe-agent-traj",
        "environment": data.get("environment") or "unknown",
        "swe_agent_version": info.get("swe_agent_version") or "unknown",
        "model_name": info.get("model_name") or info.get("model") or model_name or "unknown",
    }
    return ParseResult(events=events, metrics=metrics, extra=extra)

