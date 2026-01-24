from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from sentient_evals.models import Task


class EnvironmentResources(BaseModel):
    cpus: float | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    gpus: int | None = None


class BaseEnvironmentSpec(BaseModel):
    type: str
    allow_internet: bool = True
    resources: EnvironmentResources = Field(default_factory=EnvironmentResources)
    build_timeout_sec: float | None = None


class LocalPythonEnvironmentSpec(BaseEnvironmentSpec):
    type: Literal["local_python"] = "local_python"
    entrypoint: str | None = None


class ContainerEnvironmentSpec(BaseEnvironmentSpec):
    type: Literal["container"] = "container"
    image: str | None = None


EnvironmentSpec = LocalPythonEnvironmentSpec | ContainerEnvironmentSpec


class TaskBundle(BaseModel):
    task: Task
    instruction: str
    task_dir: Path
    environment_dir: Path | None = None
    tests_dir: Path | None = None
    files_dir: Path | None = None

    env: EnvironmentSpec = Field(default_factory=LocalPythonEnvironmentSpec)
    digest: str
    extra: dict[str, Any] = Field(default_factory=dict)

