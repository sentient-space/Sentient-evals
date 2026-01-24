from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentient_evals.env import ExecResult

from .base import BaseEnvironment, EnvironmentConfig


@dataclass(frozen=True)
class _SDKRes:
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int


class DockerSDKEnvironment(BaseEnvironment):
    def __init__(
        self,
        *,
        trial_id: str,
        workspace_dir: Path,
        logs_dir: Path,
        config: EnvironmentConfig,
        environment_dir: Path | None,
        task_digest: str | None,
        image_tag_prefix: str,
    ):
        super().__init__(trial_id=trial_id, workspace_dir=workspace_dir, logs_dir=logs_dir, config=config)
        self.environment_dir = environment_dir
        self.task_digest = task_digest
        self.image_tag_prefix = image_tag_prefix
        self._client: Any | None = None
        self._container: Any | None = None
        self._image: str | None = None
        self._container_suffix = uuid.uuid4().hex[:8]

    async def start(self, *, force_build: bool = False) -> None:
        try:
            import docker  # type: ignore
        except Exception as e:
            raise RuntimeError("Install sentient-evals[docker] to use docker_sdk backend") from e

        if self.environment_dir is None:
            raise ValueError("Task bundle is missing environment/ for docker_sdk backend")
        dockerfile = self.environment_dir / "Dockerfile"
        if not dockerfile.exists():
            raise FileNotFoundError(dockerfile)

        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_running_loop()

        def _setup() -> None:
            self._client = docker.from_env()
            tag = f"{self.image_tag_prefix}:{self.task_digest or self.trial_id}"
            self._image = tag
            should_build = force_build

            if not should_build:
                try:
                    self._client.images.get(tag)
                except Exception:
                    should_build = True

            if should_build:
                self._client.images.build(
                    path=str(self.environment_dir),
                    dockerfile=str(dockerfile),
                    tag=tag,
                )

            # Cleanup any existing container with same name.
            try:
                c = self._client.containers.get(self._container_name())
                c.remove(force=True)
            except Exception:
                pass

            host_cfg = {}
            if self.config.cpus is not None:
                host_cfg["nano_cpus"] = int(float(self.config.cpus) * 1e9)
            if self.config.memory_mb is not None:
                host_cfg["mem_limit"] = f"{int(self.config.memory_mb)}m"

            network_mode = "none" if not self.config.allow_internet else None

            self._container = self._client.containers.run(
                tag,
                ["sh", "-lc", "tail -f /dev/null"],
                name=self._container_name(),
                detach=True,
                working_dir="/workspace",
                volumes={
                    str(self.workspace_dir): {"bind": "/workspace", "mode": "rw"},
                    str(self.logs_dir): {"bind": "/logs", "mode": "rw"},
                },
                network_mode=network_mode,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                **host_cfg,
            )

        await loop.run_in_executor(None, _setup)

    async def stop(self, *, delete: bool = True) -> None:
        loop = asyncio.get_running_loop()

        def _stop():
            if self._container is not None:
                try:
                    if delete:
                        self._container.remove(force=True)
                    else:
                        self._container.stop()
                except Exception:
                    pass
            self._container = None

        await loop.run_in_executor(None, _stop)

    async def exec(self, cmd: str, *, timeout_s: float | None = None) -> ExecResult:
        if self._container is None:
            raise RuntimeError("Container not started")
        loop = asyncio.get_running_loop()

        def _run() -> _SDKRes:
            started = loop.time()
            full = ["sh", "-lc", cmd]
            r = self._container.exec_run(full, workdir="/workspace", demux=True)
            dur_ms = int((loop.time() - started) * 1000)
            out = r.output or (b"", b"")
            stdout_b, stderr_b = out if isinstance(out, tuple) else (out, b"")
            return _SDKRes(
                stdout=(stdout_b or b"").decode("utf-8", errors="replace"),
                stderr=(stderr_b or b"").decode("utf-8", errors="replace"),
                exit_code=int(getattr(r, "exit_code", 0)),
                duration_ms=dur_ms,
            )

        r = await loop.run_in_executor(None, _run)
        return ExecResult(stdout=r.stdout, stderr=r.stderr, exit_code=r.exit_code, duration_ms=r.duration_ms)

    async def upload_file(self, source_path: Path, target_path: str) -> None:
        target = (self.workspace_dir / target_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_path.read_bytes())

    async def upload_dir(self, source_dir: Path, target_dir: str) -> None:
        dst = (self.workspace_dir / target_dir).resolve()
        dst.mkdir(parents=True, exist_ok=True)
        for p in sorted(source_dir.rglob("*")):
            rel = p.relative_to(source_dir)
            out = dst / rel
            if p.is_dir():
                out.mkdir(parents=True, exist_ok=True)
            elif p.is_file() and not p.is_symlink():
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(p.read_bytes())

    async def download_file(self, source_path: str, target_path: Path) -> None:
        src = (self.workspace_dir / source_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(src.read_bytes())

    async def download_dir(self, source_dir: str, target_dir: Path) -> None:
        src = (self.workspace_dir / source_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(src.rglob("*")):
            rel = p.relative_to(src)
            out = target_dir / rel
            if p.is_dir():
                out.mkdir(parents=True, exist_ok=True)
            elif p.is_file() and not p.is_symlink():
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(p.read_bytes())

    def _container_name(self) -> str:
        # Docker names are stricter than trial_id; reuse docker_cli sanitization logic.
        safe = "".join([c if c.isalnum() or c in "_.-" else "_" for c in self.trial_id])
        return f"sentient-evals__{safe[:120]}__{self._container_suffix}"

