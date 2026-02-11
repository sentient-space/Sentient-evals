from __future__ import annotations

import asyncio
import atexit
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sentient_evals.env import ExecResult

from .base import CloudSandboxProvider, SandboxCapabilities, SandboxCreateParams, SandboxResources


class _DaytonaClientManager:
    _instance: "_DaytonaClientManager | None" = None
    _lock = asyncio.Lock()

    def __init__(self) -> None:
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

    async def get(self) -> Any:
        async with self._client_lock:
            if self._client is None:
                from daytona import AsyncDaytona  

                self._client = AsyncDaytona()
                if not self._cleanup_registered:
                    atexit.register(self._cleanup_sync)
                    self._cleanup_registered = True
            return self._client

    def _cleanup_sync(self) -> None:
        try:
            asyncio.run(self._cleanup())
        except Exception:
            pass

    async def _cleanup(self) -> None:
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


def _chunked(items: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class DaytonaProvider(CloudSandboxProvider):
    name = "daytona"
    capabilities = SandboxCapabilities(
        supports_network_toggle=True,
        supports_snapshots=True,
        supports_gpus=False,
        supports_attach=True,
        supports_stop=True,
        max_upload_batch=200,
    )

    async def create(self, params: SandboxCreateParams) -> Any:
        from daytona import (  
            CreateSandboxFromImageParams,
            CreateSandboxFromSnapshotParams,
            Image,
            Resources,
        )

        mgr = await _DaytonaClientManager.instance()
        daytona = await mgr.get()

        snapshot = params.snapshot
        if snapshot and not params.force_build:
            try:
                return await daytona.create(
                    CreateSandboxFromSnapshotParams(
                        snapshot=snapshot,
                        auto_delete_interval=0,
                        auto_stop_interval=0,
                        network_block_all=params.network_block_all,
                    ),
                    timeout=round(params.build_timeout_sec or 0),
                )
            except Exception:
                if params.dockerfile is None or params.context_dir is None:
                    raise
                await self._create_snapshot(snapshot, params)
                return await daytona.create(
                    CreateSandboxFromSnapshotParams(
                        snapshot=snapshot,
                        auto_delete_interval=0,
                        auto_stop_interval=0,
                        network_block_all=params.network_block_all,
                    ),
                    timeout=round(params.build_timeout_sec or 0),
                )

        if params.image is None and params.dockerfile is None:
            raise RuntimeError(
                "Daytona provider requires snapshot, image, or dockerfile to create a sandbox."
            )

        image = params.image
        if image is None and params.dockerfile is not None:
            image = Image.from_dockerfile(params.dockerfile)

        resources = None
        if params.resources:
            mem_gb = None
            if params.resources.memory_mb is not None:
                mem_gb = max(1, int((int(params.resources.memory_mb) + 1023) / 1024))
            resources = Resources(
                cpu=params.resources.cpus,
                memory=mem_gb,
                disk=None if params.resources.storage_mb is None else max(1, int((params.resources.storage_mb + 1023) / 1024)),
            )

        return await daytona.create(
            CreateSandboxFromImageParams(
                image=image,
                resources=resources,
                auto_delete_interval=0,
                auto_stop_interval=0,
                network_block_all=params.network_block_all,
            ),
            timeout=round(params.build_timeout_sec or 0),
        )

    async def delete(self, sandbox: Any) -> None:
        await sandbox.delete()

    async def stop(self, sandbox: Any) -> None:
        if hasattr(sandbox, "stop"):
            await sandbox.stop()
        else:
            await sandbox.delete()

    async def exec(
        self,
        sandbox: Any,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        loop = asyncio.get_running_loop()
        started = loop.time()
        resp = await sandbox.process.exec(  
            command=command,
            cwd=cwd,
            env=env,
            timeout=int(timeout_s) if timeout_s is not None else None,
        )
        dur_ms = int((loop.time() - started) * 1000)
        ex = _extract_exec(resp)
        return ExecResult(stdout=ex.stdout, stderr=ex.stderr, exit_code=ex.exit_code, duration_ms=dur_ms)

    async def upload_file(self, sandbox: Any, source_path: Path, target_path: str) -> None:
        await sandbox.fs.create_folder(str(Path(target_path).parent), "755") 
        await sandbox.fs.upload_file(str(source_path), target_path) 

    async def upload_dir(self, sandbox: Any, source_dir: Path, target_dir: str) -> None:
        from daytona import FileUpload 

        await sandbox.fs.create_folder(target_dir, "755") 
        files: list[FileUpload] = []
        for p in sorted(source_dir.rglob("*")):
            if p.is_file() and not p.is_symlink():
                rel = p.relative_to(source_dir).as_posix()
                files.append(FileUpload(source=str(p), destination=f"{target_dir}/{rel}"))
        if not files:
            return
        batch = self.capabilities.max_upload_batch or len(files)
        for chunk in _chunked(files, batch):
            await sandbox.fs.upload_files(files=chunk) 

    async def download_file(self, sandbox: Any, source_path: str, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        await sandbox.fs.download_file(source_path, str(target_path))  

    async def download_dir(self, sandbox: Any, source_dir: str, target_dir: Path) -> None:
        from daytona import FileDownloadRequest  

        target_dir.mkdir(parents=True, exist_ok=True)
        search_result = await sandbox.fs.search_files(source_dir, "*") 

        file_downloads = []
        for file_path in search_result.files:
            file_info = await sandbox.fs.get_file_info(file_path)  
            if getattr(file_info, "is_dir", False):
                continue
            path_obj = Path(file_path)
            relative_path = path_obj.relative_to(Path(source_dir))
            local_file_path = target_dir / relative_path
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            file_downloads.append(
                FileDownloadRequest(
                    source=file_path,
                    destination=str(local_file_path),
                )
            )
        if file_downloads:
            await sandbox.fs.download_files(files=file_downloads) 

    async def _create_snapshot(self, snapshot_name: str, params: SandboxCreateParams) -> None:
        if params.dockerfile is None or params.context_dir is None:
            raise RuntimeError("Snapshot build requires dockerfile and context_dir")
        if shutil.which("daytona") is None:
            raise RuntimeError("daytona CLI not found; cannot build snapshot from Dockerfile")
        cmd = [
            "daytona",
            "snapshot",
            "create",
            snapshot_name,
            "--dockerfile",
            str(params.dockerfile),
            "--context",
            str(params.context_dir),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
