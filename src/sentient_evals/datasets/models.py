"""Pydantic models for dataset registry (Harbor-compatible)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class RegistryTaskId(BaseModel):
    """Reference to a task bundle in a git repository."""

    name: str
    git_url: str
    git_commit_id: str | None = None
    path: str

    def get_cache_key(self) -> str:
        commit = self.git_commit_id or "HEAD"
        return f"{self.git_url}@{commit}:{self.path}"


class MetricConfig(BaseModel):
    """Metric configuration for aggregating dataset scores."""

    type: str
    kwargs: dict = Field(default_factory=dict)


class DatasetSpec(BaseModel):
    """Dataset specification (Harbor-compatible JSON schema)."""

    name: str
    version: str
    description: str = ""
    tasks: list[RegistryTaskId] = Field(default_factory=list)
    metrics: list[MetricConfig] = Field(default_factory=list)

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    def get_qualified_name(self) -> str:
        return f"{self.name}@{self.version}"


class Registry(BaseModel):
    """Registry containing multiple datasets."""

    name: str | None = None
    url: str | None = None
    path: Path | None = None
    datasets: list[DatasetSpec] = Field(default_factory=list)


class DownloadedTask(BaseModel):
    """Result of downloading a task."""

    task_id: RegistryTaskId
    local_path: Path
