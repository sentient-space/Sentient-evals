from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


EvalDatasetKind = Literal[
    "task_bundles",
    "registry",
    "json_tasks",
    "trace_query",
    "connector_snapshot",
    "connector_live",
]
RunTarget = Literal["local", "hosted", "customer"]


class EvalDefinitionMetadata(BaseModel):
    id: str | None = None
    name: str | None = None
    version: str = "draft"
    description: str = ""
    labels: list[str] = Field(default_factory=list)


class EvalDefinitionDataset(BaseModel):
    kind: EvalDatasetKind = "task_bundles"
    path: str | None = None
    registry_ref: str | None = None
    source: dict[str, Any] = Field(default_factory=dict)


class EmbeddedTaskDefinition(BaseModel):
    id: str
    name: str
    instruction: str
    expected_output: str | None = None
    task_config: dict[str, Any] = Field(default_factory=dict)
    test_script: str | None = None
    environment_spec: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float | None = None


class EvalDefinitionExecution(BaseModel):
    evaluators: list[dict[str, Any]] = Field(default_factory=list)
    suite_config: dict[str, Any] = Field(default_factory=dict)
    env: str | None = None
    supported_run_targets: list[RunTarget] = Field(default_factory=lambda: ["local"])


class EvalDefinitionProvenance(BaseModel):
    schema_version: str = "v1alpha1"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    harness_version: str | None = None
    git_sha: str | None = None
    connector_snapshot: dict[str, Any] | None = None


class UnifiedEvalDefinition(BaseModel):
    schema_version: Literal["v1alpha1"] = "v1alpha1"
    metadata: EvalDefinitionMetadata = Field(default_factory=EvalDefinitionMetadata)
    dataset: EvalDefinitionDataset = Field(default_factory=EvalDefinitionDataset)
    embedded_tasks: list[EmbeddedTaskDefinition] = Field(default_factory=list)
    execution: EvalDefinitionExecution = Field(default_factory=EvalDefinitionExecution)
    provenance: EvalDefinitionProvenance = Field(default_factory=EvalDefinitionProvenance)
    sentient: dict[str, Any] = Field(default_factory=dict)
