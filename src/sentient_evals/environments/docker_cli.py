from __future__ import annotations

import asyncio
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
import re
import uuid

from sentient_evals.env import ExecResult

from .base import BaseEnvironment, EnvironmentConfig


@dataclass(frozen=True)
class _HostExecResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int


async def _host_exec(cmd: str, *, cwd: Path | None = None, timeout_s: float | None = None) -> _HostExecResult:
    loop = asyncio.get_running_loop()

    def _run() -> _HostExecResult:
        started = loop.time()
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        dur_ms = int((loop.time() - started) * 1000)
        return _HostExecResult(proc.stdout, proc.stderr, int(proc.returncode), dur_ms)

    return await loop.run_in_executor(None, _run)


class DockerCLIEnvironment(BaseEnvironment):
    def __init__(
        self,
        *,
        trial_id: str,
        workspace_dir: Path,
        logs_dir: Path,
        config: EnvironmentConfig,
        engine: str,
        environment_dir: Path | None,
        task_digest: str | None,
        image_tag_prefix: str,
    ):
        super().__init__(trial_id=trial_id, workspace_dir=workspace_dir, logs_dir=logs_dir, config=config)
        self.engine = engine
        self.environment_dir = environment_dir
        self.task_digest = task_digest
        self.image_tag_prefix = image_tag_prefix
        unique = uuid.uuid4().hex[:8]
        self._container_name = f"sentient-evals__{self._sanitize_id(trial_id)}__{unique}"
        self._image = None

    async def start(self, *, force_build: bool = False) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        image = await self._ensure_image(force_build=force_build)
        self._image = image

        await _host_exec(f"{self.engine} rm -f {shlex.quote(self._container_name)} >/dev/null 2>&1 || true")

        mounts = [
            f"-v {shlex.quote(str(self.workspace_dir))}:/workspace",
            f"-v {shlex.quote(str(self.logs_dir))}:/logs",
        ]
        opts = ["-d", "--name", shlex.quote(self._container_name)]
        if self.config.cpus is not None:
            opts += ["--cpus", shlex.quote(str(self.config.cpus))]
        if self.config.memory_mb is not None:
            opts += ["--memory", shlex.quote(f"{int(self.config.memory_mb)}m")]
        if not self.config.allow_internet:
            opts += ["--network", "none"]
        # Conservative hardening: drop Linux capabilities and prevent privilege escalation.
        # Keep rootfs writable because the workspace is bind-mounted.
        if self.engine == "docker":
            opts += ["--cap-drop", "ALL", "--security-opt", "no-new-privileges"]

        cmd = " ".join(
            [
                self.engine,
                "run",
                *opts,
                *mounts,
                shlex.quote(image),
                "sh",
                "-lc",
                "tail -f /dev/null",
            ]
        )
        res = await _host_exec(cmd, timeout_s=self.config.build_timeout_sec)
        if res.exit_code != 0:
            raise RuntimeError(res.stderr.strip() or "failed to start container")

    async def stop(self, *, delete: bool = True) -> None:
        if delete:
            await _host_exec(f"{self.engine} rm -f {shlex.quote(self._container_name)} >/dev/null 2>&1 || true")
        else:
            await _host_exec(f"{self.engine} stop {shlex.quote(self._container_name)} >/dev/null 2>&1 || true")

    async def exec(self, cmd: str, *, timeout_s: float | None = None) -> ExecResult:
        qcmd = shlex.quote(cmd)
        full = (
            f"{self.engine} exec -w /workspace {shlex.quote(self._container_name)} "
            f"sh -lc {qcmd}"
        )
        r = await _host_exec(full, timeout_s=timeout_s)
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

    async def _ensure_image(self, *, force_build: bool) -> str:
        if self.environment_dir is None or not self.environment_dir.exists():
            raise ValueError("Task bundle is missing environment/ for container backend")
        dockerfile = self.environment_dir / "Dockerfile"
        if not dockerfile.exists():
            raise FileNotFoundError(dockerfile)

        tag = None
        if self.task_digest:
            tag = f"{self.image_tag_prefix}:{self.task_digest}"
        else:
            tag = f"{self.image_tag_prefix}:{self.trial_id}"

        if not force_build:
            insp = await _host_exec(f"{self.engine} image inspect {shlex.quote(tag)} >/dev/null 2>&1 || true")
            if insp.exit_code == 0:
                return tag

        build_timeout = self.config.build_timeout_sec
        cmd = f"{self.engine} build -t {shlex.quote(tag)} -f {shlex.quote(str(dockerfile))} {shlex.quote(str(self.environment_dir))}"
        res = await _host_exec(cmd, timeout_s=build_timeout)
        if res.exit_code != 0:
            raise RuntimeError(res.stderr.strip() or "failed to build image")
        return tag

    def _sanitize_id(self, s: str) -> str:
        s = s.strip()
        s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
        return s[:120] if len(s) > 120 else s

