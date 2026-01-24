from __future__ import annotations

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

    async def start(self, *, force_build: bool = False) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if self._task_files_dir and self._task_files_dir.exists():
            self._copy_tree(self._task_files_dir, self.workspace_dir)

    async def stop(self, *, delete: bool = True) -> None:
        return None

    async def exec(self, cmd: str, *, timeout_s: float | None = None):
        return await self._executor.exec(cmd, timeout_s=timeout_s)

    async def upload_file(self, source_path: Path, target_path: str) -> None:
        p = (self.workspace_dir / target_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, p)

    async def upload_dir(self, source_dir: Path, target_dir: str) -> None:
        self._copy_tree(source_dir, (self.workspace_dir / target_dir).resolve())

    async def download_file(self, source_path: str, target_path: Path) -> None:
        src = (self.workspace_dir / source_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path) -> None:
        src = (self.workspace_dir / source_dir).resolve()
        self._copy_tree(src, target_dir)

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

