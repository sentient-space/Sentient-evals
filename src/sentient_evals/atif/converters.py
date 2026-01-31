from __future__ import annotations

import json
from typing import Any, Sequence

from sentient_evals.atif.models import (
    AgentSchema,
    FinalMetricsSchema,
    MetricsSchema,
    ObservationResultSchema,
    ObservationSchema,
    StepSchema,
    ToolCallSchema,
    TrajectorySchema,
)
from sentient_evals.models import TranscriptEvent


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _source_from_role(role: str | None) -> str:
    if role == "assistant":
        return "agent"
    if role == "user":
        return "user"
    return "system"


def _build_metrics(payload: dict[str, Any]) -> MetricsSchema:
    known_fields = set(MetricsSchema.model_fields.keys())
    known: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in payload.items():
        if key in known_fields and key != "extra":
            known[key] = value
        else:
            extra[key] = value
    existing_extra = payload.get("extra")
    if isinstance(existing_extra, dict):
        extra = {**existing_extra, **extra}
    if extra:
        known["extra"] = extra
    # Coerce token metrics to ints (ATIF types) while keeping cost as float.
    for k in ("prompt_tokens", "completion_tokens", "cached_tokens"):
        if k in known and known[k] is not None:
            try:
                known[k] = int(known[k])
            except Exception:
                known.pop(k, None)
    if "cost_usd" in known and known["cost_usd"] is not None:
        try:
            known["cost_usd"] = float(known["cost_usd"])
        except Exception:
            known.pop("cost_usd", None)
    return MetricsSchema(**known)


def transcript_to_trajectory(
    transcript: Sequence[TranscriptEvent],
    *,
    session_id: str,
    agent_name: str,
    agent_version: str,
    model_name: str | None = None,
    tool_definitions: list[dict[str, Any]] | None = None,
    notes: str | None = None,
    extra: dict[str, Any] | None = None,
) -> TrajectorySchema:
    steps: list[StepSchema] = []
    for idx, ev in enumerate(transcript, start=1):
        timestamp = ev.t.isoformat() if ev.t else None
        metrics = _build_metrics(ev.metrics) if ev.metrics else None
        if ev.kind == "message":
            steps.append(
                StepSchema(
                    step_id=idx,
                    timestamp=timestamp,
                    source=_source_from_role(ev.role),
                    model_name=model_name,
                    message=ev.content or "",
                    reasoning_content=ev.reasoning_content,
                    metrics=metrics,
                    extra=ev.extra,
                )
            )
            continue

        if ev.kind == "tool_call" and ev.tool_call is not None:
            call_id = ev.tool_call.id or f"call-{idx}"
            tool_call = ToolCallSchema(
                tool_call_id=call_id,
                function_name=ev.tool_call.name,
                arguments=ev.tool_call.args or {},
            )
            observation = None
            if ev.observation is not None:
                observation = ObservationSchema(
                    results=[
                        ObservationResultSchema(
                            source_call_id=call_id, content=_stringify(ev.observation)
                        )
                    ]
                )
            steps.append(
                StepSchema(
                    step_id=idx,
                    timestamp=timestamp,
                    source="agent",
                    model_name=model_name,
                    message=ev.content or f"Tool call {ev.tool_call.name}",
                    reasoning_content=ev.reasoning_content,
                    tool_calls=[tool_call],
                    observation=observation,
                    metrics=metrics,
                    extra=ev.extra,
                )
            )
            continue

        if ev.kind == "observation":
            observation = ObservationSchema(
                results=[ObservationResultSchema(content=_stringify(ev.observation))]
            )
            steps.append(
                StepSchema(
                    step_id=idx,
                    timestamp=timestamp,
                    source="system",
                    model_name=model_name,
                    message=ev.content or "observation",
                    observation=observation,
                    metrics=metrics,
                    extra=ev.extra,
                )
            )
            continue

        if ev.kind == "metric":
            steps.append(
                StepSchema(
                    step_id=idx,
                    timestamp=timestamp,
                    source="system",
                    model_name=model_name,
                    message=ev.content or "metric",
                    metrics=metrics,
                    extra=ev.extra,
                )
            )
            continue

        steps.append(
            StepSchema(
                step_id=idx,
                timestamp=timestamp,
                source="system",
                model_name=model_name,
                message=ev.content or "",
                metrics=metrics,
                extra=ev.extra,
            )
        )

    agent = AgentSchema(
        name=agent_name,
        version=agent_version,
        model_name=model_name,
        tool_definitions=tool_definitions,
    )

    total_prompt = 0
    total_completion = 0
    total_cached = 0
    total_cost = 0.0
    any_cost = False
    for step in steps:
        if not step.metrics:
            continue
        if step.metrics.prompt_tokens is not None:
            total_prompt += int(step.metrics.prompt_tokens)
        if step.metrics.completion_tokens is not None:
            total_completion += int(step.metrics.completion_tokens)
        if step.metrics.cached_tokens is not None:
            total_cached += int(step.metrics.cached_tokens)
        if step.metrics.cost_usd is not None:
            total_cost += float(step.metrics.cost_usd)
            any_cost = True
    final_metrics = None
    if total_prompt or total_completion or total_cached or any_cost:
        final_metrics = FinalMetricsSchema(
            total_prompt_tokens=total_prompt or None,
            total_completion_tokens=total_completion or None,
            total_cached_tokens=total_cached or None,
            total_cost_usd=total_cost if any_cost else None,
            total_steps=len(steps) if steps else None,
        )
    return TrajectorySchema(
        session_id=session_id,
        agent=agent,
        steps=steps,
        notes=notes,
        final_metrics=final_metrics,
        extra=extra,
    )
