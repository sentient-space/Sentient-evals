from __future__ import annotations

import asyncio
import atexit
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentient_evals.env import ExecResult

from .base import BaseEnvironment, EnvironmentConfig


class _DaytonaClientManager:
    _instance: "_DaytonaClientManager | None" = None
    _lock = asyncio.Lock()

    def __init__(self):
        self._client: Any | None = None
        self._client_lock = asyncio.Lock()
        self._cleanup_registered = False

    @classmethod
    async def instance(cls) -> "_DaytonaClientManager":
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def get(self):
        async with self._client_lock:
            if self._client is None:
                from daytona import AsyncDaytona  # type: ignore

                self._client = AsyncDaytona()
                if not self._cleanup_registered:
                    atexit.register(self._cleanup_sync)
                    self._cleanup_registered = True
            return self._client

    def _cleanup_sync(self):
        try:
            asyncio.run(self._cleanup())
        except Exception:
            pass

    async def _cleanup(self):
        async with self._client_lock:
            if self._client is not None:
                try:
                    await self._client.close()
                except Exception:
                    pass
                self._client = None


@dataclass(frozen=True)
class _ExecExtracted:
    exit_code: int
    stdout: str
    stderr: str


def _extract_exec(resp: Any) -> _ExecExtracted:
    exit_code = getattr(resp, "exit_code", None)
    stdout = getattr(resp, "stdout", None)
    stderr = getattr(resp, "stderr", None)
    if exit_code is None:
        exit_code = getattr(resp, "return_code", 0)
    if stdout is None and hasattr(resp, "result"):
        stdout = str(getattr(resp, "result") or "")
    if stdout is None:
        stdout = ""
    if stderr is None:
        stderr = ""
    return _ExecExtracted(int(exit_code), str(stdout), str(stderr))


class DaytonaEnvironment(BaseEnvironment):
    def __init__(
        self,
        *,
        trial_id: str,
        workspace_dir: Path,
        logs_dir: Path,
        config: EnvironmentConfig,
        environment_dir: Path | None,
        task_digest: str | None,
        image: str | None,
        snapshot_template_name: str | None = None,
        network_block_all: bool | None = None,
    ):
        super().__init__(trial_id=trial_id, workspace_dir=workspace_dir, logs_dir=logs_dir, config=config)
        self.environment_dir = environment_dir
        self.task_digest = task_digest
        self.image = image
        self.snapshot_template_name = snapshot_template_name
        self.network_block_all = network_block_all
        self._sandbox: Any | None = None
        self._work_dir: str | None = None
        self._workspace_remote: str = "/workspace"
        self._logs_remote: str = "/logs"

    async def start(self, *, force_build: bool = False) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        from daytona import (  # type: ignore
            CreateSandboxFromImageParams,
            CreateSandboxFromSnapshotParams,
            Resources,
        )

        mgr = await _DaytonaClientManager.instance()
        daytona = await mgr.get()

        snapshot_name = None
        if self.snapshot_template_name and self.task_digest:
            snapshot_name = self.snapshot_template_name.format(name=self.task_digest[:12])

        sandbox = None
        if snapshot_name:
            try:
                sandbox = await daytona.create(CreateSandboxFromSnapshotParams(snapshot=snapshot_name), timeout=0)
            except Exception:
                await self._maybe_create_snapshot(snapshot_name)
                sandbox = await daytona.create(CreateSandboxFromSnapshotParams(snapshot=snapshot_name), timeout=0)
        else:
            if not self.image:
                raise RuntimeError(
                    "Daytona backend requires either a snapshot template (to build from Dockerfile via CLI) "
                    "or environment.image set in task.toml."
                )

        if sandbox is None:
            resources = None
            if self.config.cpus is not None or self.config.memory_mb is not None:
                # Daytona Resources.memory is in GB per docs; round up from MB.
                mem_gb = None
                if self.config.memory_mb is not None:
                    mem_gb = max(1, int((int(self.config.memory_mb) + 1023) / 1024))
                cpu = self.config.cpus
                resources = Resources(cpu=cpu, memory=mem_gb)  # type: ignore[arg-type]

            sandbox = await daytona.create(
                CreateSandboxFromImageParams(
                    image=self.image,
                    resources=resources,
                ),
                timeout=0,
            )

        self._sandbox = sandbox
        self._work_dir = await sandbox.get_work_dir()

        await sandbox.fs.create_folder(self._workspace_remote, "755")
        await sandbox.fs.create_folder(self._logs_remote, "755")

    async def stop(self, *, delete: bool = True) -> None:
        if self._sandbox is None:
            return
        try:
            try:
                await self.download_dir(self._logs_remote, self.logs_dir)
            except Exception:
                # best-effort
                pass
        finally:
            try:
                if delete:
                    await self._sandbox.delete()
                else:
                    await self._sandbox.stop()
            except Exception:
                pass
            self._sandbox = None

    async def exec(self, cmd: str, *, timeout_s: float | None = None) -> ExecResult:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")
        loop = asyncio.get_running_loop()
        started = loop.time()
        resp = await self._sandbox.process.exec(  # type: ignore[attr-defined]
            command=cmd,
            cwd=self._workspace_remote,
            env=None,
            timeout=int(timeout_s) if timeout_s is not None else None,
        )
        dur_ms = int((loop.time() - started) * 1000)
        ex = _extract_exec(resp)
        return ExecResult(stdout=ex.stdout, stderr=ex.stderr, exit_code=ex.exit_code, duration_ms=dur_ms)

    async def upload_file(self, source_path: Path, target_path: str) -> None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")
        remote = self._remote_path(target_path)
        await self._sandbox.fs.create_folder(str(Path(remote).parent), "755")  # type: ignore[attr-defined]
        await self._sandbox.fs.upload_file(str(source_path), remote)  # type: ignore[attr-defined]

    async def upload_dir(self, source_dir: Path, target_dir: str) -> None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")
        base_remote = self._remote_path(target_dir)
        await self._sandbox.fs.create_folder(base_remote, "755")  # type: ignore[attr-defined]

        from daytona import FileUpload  # type: ignore

        files: list[FileUpload] = []
        for p in sorted(source_dir.rglob("*")):
            if p.is_file() and not p.is_symlink():
                rel = p.relative_to(source_dir).as_posix()
                files.append(FileUpload(source=str(p), destination=f"{base_remote}/{rel}"))
        if files:
            await self._sandbox.fs.upload_files(files)  # type: ignore[attr-defined]

    async def download_file(self, source_path: str, target_path: Path) -> None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        remote = self._remote_path(source_path)
        await self._sandbox.fs.download_file(remote, str(target_path))  # type: ignore[attr-defined]

    async def download_dir(self, source_dir: str, target_dir: Path) -> None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")
        target_dir.mkdir(parents=True, exist_ok=True)
        remote = self._remote_path(source_dir)
        await self._download_dir_recursive(remote, target_dir)

    async def _download_dir_recursive(self, remote_dir: str, local_dir: Path) -> None:
        files = await self._sandbox.fs.list_files(remote_dir)  # type: ignore[attr-defined]
        for f in files:
            name = getattr(f, "name", None) or getattr(f, "path", None) or ""
            is_dir = bool(getattr(f, "is_dir", False))
            if not name:
                continue
            r = f"{remote_dir.rstrip('/')}/{name}"
            out = local_dir / name
            if is_dir:
                out.mkdir(parents=True, exist_ok=True)
                await self._download_dir_recursive(r, out)
            else:
                await self._sandbox.fs.download_file(r, str(out))  # type: ignore[attr-defined]

    def _remote_path(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        return f"{self._workspace_remote.rstrip('/')}/{p}"

    async def _maybe_create_snapshot(self, snapshot_name: str) -> None:
        if self.environment_dir is None:
            raise RuntimeError("Task bundle is missing environment/ for snapshot build")
        dockerfile = self.environment_dir / "Dockerfile"
        if not dockerfile.exists():
            raise FileNotFoundError(dockerfile)

        if shutil.which("daytona") is None:
            raise RuntimeError("daytona CLI not found; cannot build snapshot from Dockerfile")

        cmd = [
            "daytona",
            "snapshot",
            "create",
            snapshot_name,
            "--dockerfile",
            str(dockerfile),
            "--context",
            str(self.environment_dir),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

