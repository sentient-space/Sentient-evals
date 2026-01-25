from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ATIF_VERSION = "ATIF-v1.5"


class SubagentTrajectoryRefSchema(BaseModel):
    session_id: str
    trajectory_path: str | None = None
    extra: dict[str, Any] | None = None


class ObservationResultSchema(BaseModel):
    source_call_id: str | None = None
    content: str | None = None
    subagent_trajectory_ref: list[SubagentTrajectoryRefSchema] | None = None


class ObservationSchema(BaseModel):
    results: list[ObservationResultSchema] = Field(default_factory=list)


class ToolCallSchema(BaseModel):
    tool_call_id: str
    function_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class MetricsSchema(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    extra: dict[str, Any] | None = None


class StepSchema(BaseModel):
    step_id: int
    timestamp: str | None = None
    source: Literal["system", "user", "agent"]
    model_name: str | None = None
    reasoning_effort: str | float | None = None
    message: str
    reasoning_content: str | None = None
    tool_calls: list[ToolCallSchema] | None = None
    observation: ObservationSchema | None = None
    metrics: MetricsSchema | None = None
    extra: dict[str, Any] | None = None


class AgentSchema(BaseModel):
    name: str
    version: str
    model_name: str | None = None
    tool_definitions: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None


class FinalMetricsSchema(BaseModel):
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_cached_tokens: int | None = None
    total_cost_usd: float | None = None
    total_steps: int | None = None
    extra: dict[str, Any] | None = None


class TrajectorySchema(BaseModel):
    schema_version: str = ATIF_VERSION
    session_id: str
    agent: AgentSchema
    steps: list[StepSchema] = Field(default_factory=list)
    notes: str | None = None
    final_metrics: FinalMetricsSchema | None = None
    continued_trajectory_ref: str | None = None
    extra: dict[str, Any] | None = None
