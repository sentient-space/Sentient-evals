from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)

SCHEMA_VERSION = "v1"


class Severity(str, Enum):
    info = "info"
    warning = "warning"
    error = "error"


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    error: str | None = None


class OutcomePointer(BaseModel):
    """
    Pointer to an external/large artifact describing the final environment state.

    Examples: a tarball of a workspace, a DB snapshot key, an S3 URI, etc.
    """

    kind: Literal["file", "dir", "uri", "s3", "db"] = "uri"
    ref: str
    digest: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TranscriptEvent(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    t: datetime = Field(default_factory=utcnow)
    kind: Literal["message", "tool_call", "observation", "metric"]
    role: Literal["system", "user", "assistant"] | None = None
    content: str | None = None
    tool_call: ToolCall | None = None
    observation: Any | None = None
    metrics: dict[str, float] | None = None


class Outcome(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    summary: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    pointers: list[OutcomePointer] = Field(default_factory=list)


class Task(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    id: str
    input: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float | None = None


class PromptSpec(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    id: str
    version: str
    content: str | None = None


class JudgeSpec(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    id: str
    version: str
    rubric: str | None = None


class GraderSpec(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    type: str
    config: dict[str, Any] = Field(default_factory=dict)
    weight: float | None = None
    required: bool = True


class TrialSpec(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    run_id: str
    trial_id: str
    task: Task
    adapter: str
    seed: int
    attempt: int
    graders: list[GraderSpec] = Field(default_factory=list)
    env: dict[str, Any] = Field(default_factory=dict)


class TrialConfig(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    run_id: str
    trial_id: str
    task_id: str
    seed: int
    attempt: int
    adapter: str
    started_at: datetime = Field(default_factory=utcnow)
    model: str | None = None
    prompt: PromptSpec | None = None
    judge: JudgeSpec | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class GraderResult(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    name: str
    score: float
    passed: bool
    severity: Severity = Severity.info
    details: dict[str, Any] = Field(default_factory=dict)


class TrialResult(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    ok: bool
    task_id: str
    trial_id: str
    adapter: str
    seed: int
    started_at: datetime
    finished_at: datetime = Field(default_factory=utcnow)
    transcript_path: str | None = None
    trajectory_path: str | None = None
    outcome_path: str | None = None
    graders: list[GraderResult] = Field(default_factory=list)
    error: str | None = None


class SuiteConfig(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    id: str = "default"
    trials_per_task: int = 1
    concurrency: int = 1
    seeds: list[int] | None = None


class RunSummary(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    run_id: str
    suite_id: str
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    task_count: int
    trial_count: int
    passed_trials: int
    failed_trials: int
    avg_score: float | None = None


class RunStatus(str, Enum):
    running = "running"
    completed = "completed"
    cancelled = "cancelled"
    failed = "failed"


class RunResult(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    run_id: str
    suite_id: str
    status: RunStatus
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None

    task_count: int
    trial_count: int
    passed_trials: int
    failed_trials: int
    avg_score: float | None = None

    metrics: dict[str, Any] = Field(default_factory=dict)
    per_task: dict[str, Any] = Field(default_factory=dict)


class RunConfigFile(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    run_id: str
    suite: SuiteConfig
    adapter: str
    started_at: datetime = Field(default_factory=utcnow)
    harness_version: str
    model: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

