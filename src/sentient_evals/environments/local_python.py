from __future__ import annotations

import re
import shutil
from pathlib import Path

from sentient_evals.env import LocalToolExecutor

from .base import BaseEnvironment, EnvironmentConfig


class LocalPythonEnvironment(BaseEnvironment):
    def __init__(
        self,
        *,
        trial_id: str,
        workspace_dir: Path,
        logs_dir: Path,
        config: EnvironmentConfig,
        task_files_dir: Path | None = None,
    ):
        super().__init__(trial_id=trial_id, workspace_dir=workspace_dir, logs_dir=logs_dir, config=config)
        self._executor = LocalToolExecutor(root=workspace_dir)
        self._task_files_dir = task_files_dir
        self._tests_dir = self.logs_dir / "tests"

    async def start(self, *, force_build: bool = False) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._tests_dir.mkdir(parents=True, exist_ok=True)
        if self._task_files_dir and self._task_files_dir.exists():
            self._copy_tree(self._task_files_dir, self.workspace_dir)

    async def stop(self, *, delete: bool = True) -> None:
        return None

    async def exec(self, cmd: str, *, timeout_s: float | None = None):
        rewritten = self._rewrite_mountpoints(cmd)
        return await self._executor.exec(rewritten, timeout_s=timeout_s)

    async def upload_file(self, source_path: Path, target_path: str) -> None:
        p = self._map_env_path(target_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, p)

    async def upload_dir(self, source_dir: Path, target_dir: str) -> None:
        self._copy_tree(source_dir, self._map_env_path(target_dir))

    async def download_file(self, source_path: str, target_path: Path) -> None:
        src = self._map_env_path(source_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path) -> None:
        src = self._map_env_path(source_dir)
        self._copy_tree(src, target_dir)

    def _map_env_path(self, path: str) -> Path:
        raw = (path or "").strip()
        if raw.startswith("/tests"):
            rel = raw[len("/tests") :].lstrip("/")
            out = (self._tests_dir / rel).resolve() if rel else self._tests_dir.resolve()
        elif raw.startswith("/logs"):
            rel = raw[len("/logs") :].lstrip("/")
            out = (self.logs_dir / rel).resolve() if rel else self.logs_dir.resolve()
        elif raw.startswith("/workspace"):
            rel = raw[len("/workspace") :].lstrip("/")
            out = (self.workspace_dir / rel).resolve() if rel else self.workspace_dir.resolve()
        else:
            out = (self.workspace_dir / raw.lstrip("/")).resolve()

        if self.workspace_dir not in out.parents and self.logs_dir not in out.parents and out not in (
            self.workspace_dir,
            self.logs_dir,
        ):
            raise ValueError("Path escapes local environment roots")
        return out

    def _rewrite_mountpoints(self, cmd: str) -> str:
        def _rewrite(prefix: str, repl: str, s: str) -> str:
            pattern = rf"(?P<pre>(^|[\s'\"`])){re.escape(prefix)}(?P<post>(/|$))"
            return re.sub(pattern, rf"\g<pre>{repl}\g<post>", s)

        s = cmd
        s = _rewrite("/tests", str(self._tests_dir), s)
        s = _rewrite("/logs", str(self.logs_dir), s)
        s = _rewrite("/workspace", str(self.workspace_dir), s)
        return s

    def _copy_tree(self, src: Path, dst: Path) -> None:
        if not src.exists():
            return
        dst.mkdir(parents=True, exist_ok=True)
        for p in sorted(src.rglob("*")):
            rel = p.relative_to(src)
            out = dst / rel
            if p.is_dir():
                out.mkdir(parents=True, exist_ok=True)
            elif p.is_file() and not p.is_symlink():
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out)

