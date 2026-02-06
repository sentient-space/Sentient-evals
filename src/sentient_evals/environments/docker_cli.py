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
        platform: str | None = None,
        workdir: str | None = None,
        workspace_mount: str | None = None,
    ):
        super().__init__(trial_id=trial_id, workspace_dir=workspace_dir, logs_dir=logs_dir, config=config)
        self.engine = engine
        self.environment_dir = environment_dir
        self.task_digest = task_digest
        self.image_tag_prefix = image_tag_prefix
        self.platform = platform
        unique = uuid.uuid4().hex[:8]
        self._container_name = f"sentient-evals__{self._sanitize_id(trial_id)}__{unique}"
        self._image = None
        self.tests_dir = workspace_dir.parent / "tests"
        self._workdir = self._normalize_path(workdir) if workdir else self._parse_workdir()
        if workspace_mount:
            self._workspace_mount = self._normalize_path(workspace_mount)
        elif self._workdir == "/testbed":
            self._workspace_mount = "/workspace"
        else:
            self._workspace_mount = self._workdir

    def _normalize_path(self, path: str | None) -> str:
        if not path:
            return ""
        if not path.startswith("/"):
            return f"/{path}"
        return path.rstrip("/") or "/"

    def _parse_workdir(self) -> str:
        """Extract WORKDIR from task Dockerfile. Defaults to /app."""
        if self.environment_dir is None:
            return "/app"
        dockerfile = self.environment_dir / "Dockerfile"
        if not dockerfile.exists():
            return "/app"
        try:
            content = dockerfile.read_text()
            matches = re.findall(r'^WORKDIR\s+([^\s#]+)', content, re.MULTILINE)
            return matches[-1].rstrip('/') if matches else "/app"
        except Exception:
            return "/app"

    async def start(self, *, force_build: bool = False) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.tests_dir.mkdir(parents=True, exist_ok=True)
        try:
            (self.logs_dir / "container_name.txt").write_text(self._container_name + "\n", encoding="utf-8")
        except Exception:
            pass

        await self._ensure_engine_available()
        image = await self._ensure_image(force_build=force_build)
        self._image = image

        await _host_exec(f"{self.engine} rm -f {shlex.quote(self._container_name)} >/dev/null 2>&1 || true")

        mounts = [
            f"-v {shlex.quote(str(self.workspace_dir))}:{self._workspace_mount}",
            f"-v {shlex.quote(str(self.logs_dir))}:/logs",
            f"-v {shlex.quote(str(self.tests_dir))}:/tests",
        ]
        opts = ["-d", "--name", shlex.quote(self._container_name)]
        if self.platform:
            opts += ["--platform", shlex.quote(self.platform)]
        if self.config.cpus is not None:
            opts += ["--cpus", shlex.quote(str(self.config.cpus))]
        if self.config.memory_mb is not None:
            opts += ["--memory", shlex.quote(f"{int(self.config.memory_mb)}m")]
        if not self.config.allow_internet:
            opts += ["--network", "none"]
        # NOTE: We intentionally do not drop all caps / set no-new-privileges by default.
        # Runtime installation flows (apt, curl installers, tar extraction) often rely on
        # setuid helpers and CAP_CHOWN. We can re-introduce hardening as an opt-in later.

        cmd = " ".join(
            [
                self.engine,
                "run",
                *opts,
                *mounts,
                shlex.quote(image),
                "sh",
                "-lc",
                shlex.quote("tail -f /dev/null"),
            ]
        )
        res = await _host_exec(cmd, timeout_s=self.config.build_timeout_sec)
        if res.exit_code != 0:
            try:
                (self.logs_dir / "docker_run_command.txt").write_text(cmd + "\n", encoding="utf-8")
                (self.logs_dir / "docker_run_stdout.txt").write_text(res.stdout or "", encoding="utf-8")
                (self.logs_dir / "docker_run_stderr.txt").write_text(res.stderr or "", encoding="utf-8")
            except Exception:
                pass
            raise RuntimeError(res.stderr.strip() or "failed to start container")

        # `docker run -d` can succeed even if the container exits immediately.
        # Verify it is actually running; if not, capture inspect/logs and raise a useful error.
        insp = await _host_exec(
            f"{self.engine} inspect {shlex.quote(self._container_name)} "
            "--format '{{.State.Status}} {{.State.ExitCode}} {{.State.Error}}'"
        )
        state = (insp.stdout or "").strip().split(" ", 1)[0] if insp.exit_code == 0 else ""
        if state and state != "running":
            try:
                (self.logs_dir / "docker_inspect.txt").write_text(
                    (insp.stdout or insp.stderr or "").strip() + "\n", encoding="utf-8"
                )
            except Exception:
                pass
            try:
                (self.logs_dir / "docker_run_command.txt").write_text(cmd + "\n", encoding="utf-8")
                (self.logs_dir / "docker_run_stdout.txt").write_text(res.stdout or "", encoding="utf-8")
                (self.logs_dir / "docker_run_stderr.txt").write_text(res.stderr or "", encoding="utf-8")
            except Exception:
                pass
            logs = await _host_exec(f"{self.engine} logs {shlex.quote(self._container_name)}")
            try:
                (self.logs_dir / "docker_logs.txt").write_text(
                    (logs.stdout or logs.stderr or "").strip() + "\n", encoding="utf-8"
                )
            except Exception:
                pass
            raise RuntimeError(
                f"container failed to stay running (state={state}). "
                f"See host logs under: {self.logs_dir}"
            )

    async def stop(self, *, delete: bool = True) -> None:
        if delete:
            try:
                insp = await _host_exec(
                    f"{self.engine} inspect {shlex.quote(self._container_name)} "
                    "--format '{{.State.Status}} {{.State.ExitCode}} {{.State.Error}}'"
                )
                (self.logs_dir / "docker_inspect_final.txt").write_text(
                    (insp.stdout or insp.stderr or "").strip() + "\n", encoding="utf-8"
                )
                logs = await _host_exec(f"{self.engine} logs {shlex.quote(self._container_name)}")
                (self.logs_dir / "docker_logs_final.txt").write_text(
                    (logs.stdout or logs.stderr or "").strip() + "\n", encoding="utf-8"
                )
            except Exception:
                pass
            await _host_exec(f"{self.engine} rm -f {shlex.quote(self._container_name)} >/dev/null 2>&1 || true")
        else:
            await _host_exec(f"{self.engine} stop {shlex.quote(self._container_name)} >/dev/null 2>&1 || true")

    async def exec(self, cmd: str, *, timeout_s: float | None = None) -> ExecResult:
        qcmd = shlex.quote(cmd)
        full = (
            f"{self.engine} exec -w {shlex.quote(self._workdir)} {shlex.quote(self._container_name)} "
            f"sh -lc {qcmd}"
        )
        r = await _host_exec(full, timeout_s=timeout_s)
        if r.exit_code != 0 and "is not running" in (r.stderr or "").lower():
            # Surface a more actionable message; callers will persist this in trial error.txt.
            raise RuntimeError(
                f"container is not running while executing command. See host logs under: {self.logs_dir}"
            )
        return ExecResult(stdout=r.stdout, stderr=r.stderr, exit_code=r.exit_code, duration_ms=r.duration_ms)

    async def upload_file(self, source_path: Path, target_path: str) -> None:
        target = (self.workspace_dir / target_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_path.read_bytes())

    async def upload_dir(self, source_dir: Path, target_dir: str) -> None:
        # Map container absolute paths to host directories
        if target_dir.startswith("/tests"):
            base = self.tests_dir
            rel_path = target_dir[6:].lstrip("/")  # Strip '/tests' prefix
            dst = (base / rel_path).resolve() if rel_path else base
        elif target_dir.startswith("/logs"):
            base = self.logs_dir
            rel_path = target_dir[5:].lstrip("/")  # Strip '/logs' prefix
            dst = (base / rel_path).resolve() if rel_path else base
        elif target_dir.startswith(self._workspace_mount):
            # Map container workspace mount to host workspace directory.
            prefix_len = len(self._workspace_mount)
            rel_path = target_dir[prefix_len:].lstrip("/")
            dst = (self.workspace_dir / rel_path).resolve() if rel_path else self.workspace_dir
        else:
            # Relative paths are relative to workspace
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
        src = self._map_container_path(source_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(src.read_bytes())

    async def download_dir(self, source_dir: str, target_dir: Path) -> None:
        src = self._map_container_path(source_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(src.rglob("*")):
            rel = p.relative_to(src)
            out = target_dir / rel
            if p.is_dir():
                out.mkdir(parents=True, exist_ok=True)
            elif p.is_file() and not p.is_symlink():
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(p.read_bytes())

    def _map_container_path(self, path: str) -> Path:
        if path.startswith("/tests"):
            base = self.tests_dir
            rel_path = path[6:].lstrip("/")
            return (base / rel_path).resolve() if rel_path else base
        if path.startswith("/logs"):
            base = self.logs_dir
            rel_path = path[5:].lstrip("/")
            return (base / rel_path).resolve() if rel_path else base
        if path.startswith(self._workspace_mount):
            prefix_len = len(self._workspace_mount)
            rel_path = path[prefix_len:].lstrip("/")
            return (self.workspace_dir / rel_path).resolve() if rel_path else self.workspace_dir
        return (self.workspace_dir / path).resolve()

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
            
            insp = await _host_exec(f"{self.engine} image inspect {shlex.quote(tag)} >/dev/null 2>&1")
            if insp.exit_code == 0: 
                return tag

        build_timeout = self.config.build_timeout_sec
        platform_flag = f" --platform {shlex.quote(self.platform)}" if self.platform else ""
        cmd = (
            f"{self.engine} build{platform_flag} -t {shlex.quote(tag)} "
            f"-f {shlex.quote(str(dockerfile))} {shlex.quote(str(self.environment_dir))}"
        )
        res = await _host_exec(cmd, timeout_s=build_timeout)
        if res.exit_code != 0:
            raise RuntimeError(res.stderr.strip() or "failed to build image")
        return tag

    async def _ensure_engine_available(self) -> None:
        res = await _host_exec(f"{self.engine} info >/dev/null 2>&1 || {self.engine} version >/dev/null 2>&1")
        if res.exit_code != 0:
            raise RuntimeError(
                f"{self.engine} is not available. Please ensure Docker/Podman is installed and running."
            )

    def _sanitize_id(self, s: str) -> str:
        s = s.strip()
        s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
        return s[:120] if len(s) > 120 else s
