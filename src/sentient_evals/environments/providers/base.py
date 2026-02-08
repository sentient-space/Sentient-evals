from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from sentient_evals.env import ExecResult


@dataclass(frozen=True)
class SandboxResources:
    cpus: float | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    gpus: int | None = None


@dataclass(frozen=True)
class SandboxCapabilities:
    supports_network_toggle: bool = False
    supports_snapshots: bool = False
    supports_gpus: bool = False
    supports_attach: bool = False
    supports_stop: bool = True
    max_upload_batch: int | None = None


@dataclass(frozen=True)
class SandboxCreateParams:
    image: str | None = None
    snapshot: str | None = None
    dockerfile: Path | None = None
    context_dir: Path | None = None
    resources: SandboxResources | None = None
    network_block_all: bool | None = None
    build_timeout_sec: float | None = None
    force_build: bool = False

    def with_force_build(self, force_build: bool) -> "SandboxCreateParams":
        return replace(self, force_build=force_build)


class CloudSandboxProvider(Protocol):
    name: str
    capabilities: SandboxCapabilities

    async def create(self, params: SandboxCreateParams) -> Any:
        ...

    async def delete(self, sandbox: Any) -> None:
        ...

    async def stop(self, sandbox: Any) -> None:
        ...

    async def exec(
        self,
        sandbox: Any,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        ...

    async def upload_file(self, sandbox: Any, source_path: Path, target_path: str) -> None:
        ...

    async def upload_dir(self, sandbox: Any, source_dir: Path, target_dir: str) -> None:
        ...

    async def download_file(self, sandbox: Any, source_path: str, target_path: Path) -> None:
        ...

    async def download_dir(self, sandbox: Any, source_dir: str, target_dir: Path) -> None:
        ...
