from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from sentient_evals.env import ExecResult


class EnvironmentType(str, Enum):
    local_python = "local_python"
    docker_cli = "docker_cli"
    docker_sdk = "docker_sdk"
    podman_cli = "podman_cli"
    daytona = "daytona"
    e2b = "e2b"


@dataclass(frozen=True)
class EnvironmentConfig:
    allow_internet: bool = True
    cpus: float | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    gpus: int | None = None
    build_timeout_sec: float | None = None
    provider_concurrency: int | None = None


class BaseEnvironment(ABC):
    def __init__(
        self,
        *,
        trial_id: str,
        workspace_dir: Path,
        logs_dir: Path,
        config: EnvironmentConfig,
    ):
        self.trial_id = trial_id
        self.workspace_dir = workspace_dir
        self.logs_dir = logs_dir
        self.config = config

    @abstractmethod
    async def start(self, *, force_build: bool = False) -> None: ...

    @abstractmethod
    async def stop(self, *, delete: bool = True) -> None: ...

    @abstractmethod
    async def exec(self, cmd: str, *, timeout_s: float | None = None) -> ExecResult: ...

    @abstractmethod
    async def upload_file(self, source_path: Path, target_path: str) -> None: ...

    @abstractmethod
    async def upload_dir(self, source_dir: Path, target_dir: str) -> None: ...

    @abstractmethod
    async def download_file(self, source_path: str, target_path: Path) -> None: ...

    @abstractmethod
    async def download_dir(self, source_dir: str, target_dir: Path) -> None: ...
