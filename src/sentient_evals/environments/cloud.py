from __future__ import annotations

import os
import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from sentient_evals.env import ExecResult

from .base import BaseEnvironment, EnvironmentConfig
from .providers.base import CloudSandboxProvider, SandboxCreateParams


@dataclass(frozen=True)
class CloudSandboxSettings:
    remote_workspace: str = "/workspace"
    remote_logs: str = "/logs"
    default_cwd: str | None = None


class CloudSandboxEnvironment(BaseEnvironment):
    def __init__(
        self,
        *,
        trial_id: str,
        workspace_dir: Path,
        logs_dir: Path,
        config: EnvironmentConfig,
        provider: CloudSandboxProvider,
        create_params: SandboxCreateParams,
        settings: CloudSandboxSettings | None = None,
    ):
        super().__init__(trial_id=trial_id, workspace_dir=workspace_dir, logs_dir=logs_dir, config=config)
        self._provider = provider
        self._create_params = create_params
        self._sandbox: Any | None = None
        self._sandbox_id: str | None = None
        self._settings = settings or CloudSandboxSettings()
        self._events_path = self.logs_dir / "env_events.jsonl"

    async def start(self, *, force_build: bool = False) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        params = self._create_params.with_force_build(force_build)
        self._sandbox = await self._call_with_retry("create_sandbox", self._provider.create, params, retries=2)
        self._sandbox_id = self._extract_sandbox_id(self._sandbox)
        self._write_env_info()
        await self._provider.exec(
            self._sandbox,
            f"mkdir -p {self._settings.remote_workspace} {self._settings.remote_logs}",
            cwd=None,
            env=None,
            timeout_s=None,
        )

    async def stop(self, *, delete: bool = True) -> None:
        if self._sandbox is None:
            return
        try:
            try:
                await self.download_dir(self._settings.remote_logs, self.logs_dir)
            except Exception:
                pass
        finally:
            try:
                if delete:
                    await self._call_with_retry(
                        "delete_sandbox", self._provider.delete, self._sandbox, retries=1
                    )
                else:
                    await self._call_with_retry(
                        "stop_sandbox", self._provider.stop, self._sandbox, retries=1
                    )
            except Exception:
                pass
            self._sandbox = None

    async def exec(self, cmd: str, *, timeout_s: float | None = None) -> ExecResult:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")
        cwd = self._settings.default_cwd or self._settings.remote_workspace
        start = time.monotonic()
        try:
            result = await self._provider.exec(
                self._sandbox,
                cmd,
                cwd=cwd,
                env=None,
                timeout_s=timeout_s,
            )
            dur = int((time.monotonic() - start) * 1000)
            self._log_event(
                "exec",
                status="ok",
                duration_ms=dur,
                exit_code=result.exit_code,
                command_digest=self._digest(cmd),
                command_len=len(cmd),
                timeout_s=timeout_s,
                cwd=cwd,
            )
            return result
        except Exception as exc:
            dur = int((time.monotonic() - start) * 1000)
            self._log_event(
                "exec",
                status="error",
                duration_ms=dur,
                error=str(exc),
                command_digest=self._digest(cmd),
                command_len=len(cmd),
                timeout_s=timeout_s,
                cwd=cwd,
            )
            raise

    async def upload_file(self, source_path: Path, target_path: str) -> None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")
        remote = self._remote_path(target_path)
        await self._call_with_retry(
            "upload_file",
            self._provider.upload_file,
            self._sandbox,
            source_path,
            remote,
            retries=2,
            file_count=1,
            bytes_total=source_path.stat().st_size if source_path.exists() else None,
        )

    async def upload_dir(self, source_dir: Path, target_dir: str) -> None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")
        remote = self._remote_path(target_dir)
        file_count, bytes_total = self._dir_stats(source_dir)
        await self._call_with_retry(
            "upload_dir",
            self._provider.upload_dir,
            self._sandbox,
            source_dir,
            remote,
            retries=2,
            file_count=file_count,
            bytes_total=bytes_total,
        )

    async def download_file(self, source_path: str, target_path: Path) -> None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")
        remote = self._remote_path(source_path)
        await self._call_with_retry(
            "download_file",
            self._provider.download_file,
            self._sandbox,
            remote,
            target_path,
            retries=2,
        )

    async def download_dir(self, source_dir: str, target_dir: Path) -> None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")
        remote = self._remote_path(source_dir)
        await self._call_with_retry(
            "download_dir",
            self._provider.download_dir,
            self._sandbox,
            remote,
            target_dir,
            retries=2,
        )

    def _remote_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return f"{self._settings.remote_workspace.rstrip('/')}/{path}"

    def _log_event(self, event: str, **payload: object) -> None:
        record = {
            "t": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "provider": self._provider.name,
            "sandbox_id": self._sandbox_id,
            **payload,
        }
        self._events_path.parent.mkdir(parents=True, exist_ok=True)
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str))
            f.write("\n")

    async def _call_with_retry(self, name: str, fn, *args, retries: int = 1, **meta):
        attempts = 0
        delay = 1.0
        while True:
            attempts += 1
            start = time.monotonic()
            try:
                result = await fn(*args)
                dur = int((time.monotonic() - start) * 1000)
                self._log_event(
                    name,
                    status="ok",
                    duration_ms=dur,
                    attempts=attempts,
                    **meta,
                )
                return result
            except Exception as exc:
                dur = int((time.monotonic() - start) * 1000)
                self._log_event(
                    name,
                    status="error",
                    duration_ms=dur,
                    attempts=attempts,
                    error=str(exc),
                    **meta,
                )
                if attempts > retries:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 8.0)

    def _write_env_info(self) -> None:
        info = {
            "provider": self._provider.name,
            "sandbox_id": self._sandbox_id,
            "image": self._create_params.image,
            "snapshot": self._create_params.snapshot,
            "network_block_all": self._create_params.network_block_all,
            "resources": self._create_params.resources.__dict__ if self._create_params.resources else None,
        }
        path = self.logs_dir / "env_info.json"
        path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def _extract_sandbox_id(sandbox: Any | None) -> str | None:
        if sandbox is None:
            return None
        for key in ("id", "sandbox_id", "uid"):
            val = getattr(sandbox, key, None)
            if val:
                return str(val)
        return None

    @staticmethod
    def _digest(text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _dir_stats(root: Path) -> tuple[int, int]:
        count = 0
        total = 0
        if not root.exists():
            return count, total
        for p in root.rglob("*"):
            if p.is_file() and not p.is_symlink():
                count += 1
                try:
                    total += p.stat().st_size
                except Exception:
                    pass
        return count, total
