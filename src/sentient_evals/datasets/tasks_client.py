"""Task client for downloading task bundles via git sparse-checkout."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

from .models import DownloadedTask, RegistryTaskId

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "sentient-evals" / "tasks"


class TaskClient:
    """Downloads task bundles from git repositories using sparse-checkout."""

    def __init__(self, cache_dir: Path | None = None):
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR

    def _get_task_cache_path(self, task: RegistryTaskId) -> Path:
        cache_key = hashlib.sha256(task.get_cache_key().encode()).hexdigest()[:12]
        return self._cache_dir / cache_key / Path(task.path).name

    def _should_download(
        self, task: RegistryTaskId, target_path: Path, overwrite: bool
    ) -> bool:
        if not target_path.exists():
            return True
        if overwrite:
            return True
        if task.git_commit_id is None:
            return True
        if not any(target_path.iterdir()):
            return True
        return False

    def _download_tasks_from_git(
        self,
        git_url: str,
        tasks: Sequence[tuple[RegistryTaskId, Path]],
    ) -> None:
        sparse_paths = {t.path for t, _ in tasks}
        head_tasks = [(t, p) for t, p in tasks if t.git_commit_id is None]
        commit_groups: dict[str, list[tuple[RegistryTaskId, Path]]] = {}
        for t, p in tasks:
            if t.git_commit_id is not None:
                commit_groups.setdefault(t.git_commit_id, []).append((t, p))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            subprocess.run(
                [
                    "git", "clone",
                    "--filter=blob:none",
                    "--depth", "1",
                    "--no-checkout",
                    git_url,
                    str(tmp_dir),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "sparse-checkout", "set", "--no-cone", *sparse_paths],
                check=True,
                capture_output=True,
                cwd=tmp_dir,
            )

            if head_tasks:
                subprocess.run(
                    ["git", "checkout"],
                    check=True,
                    capture_output=True,
                    cwd=tmp_dir,
                )
                for task, target in head_tasks:
                    self._copy_task(tmp_dir / task.path, target)

            for commit_id, commit_tasks in commit_groups.items():
                subprocess.run(
                    ["git", "fetch", "--depth", "1", "origin", commit_id],
                    check=True,
                    capture_output=True,
                    cwd=tmp_dir,
                )
                subprocess.run(
                    ["git", "checkout", commit_id],
                    check=True,
                    capture_output=True,
                    cwd=tmp_dir,
                )
                for task, target in commit_tasks:
                    self._copy_task(tmp_dir / task.path, target)

    def _copy_task(self, source: Path, target: Path) -> None:
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)

    def download_tasks(
        self,
        tasks: Sequence[RegistryTaskId],
        output_dir: Path | None = None,
        overwrite: bool = False,
    ) -> list[DownloadedTask]:
        cache_dir = output_dir or self._cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        tasks_to_download: list[tuple[RegistryTaskId, Path]] = []
        task_paths: dict[str, Path] = {}

        for task in tasks:
            target = self._get_task_cache_path(task) if output_dir is None else (
                cache_dir / Path(task.path).name
            )
            task_paths[task.name] = target
            if self._should_download(task, target, overwrite):
                tasks_to_download.append((task, target))

        by_git_url: dict[str, list[tuple[RegistryTaskId, Path]]] = {}
        for task, path in tasks_to_download:
            by_git_url.setdefault(task.git_url, []).append((task, path))

        for git_url, url_tasks in by_git_url.items():
            self._download_tasks_from_git(git_url, url_tasks)

        return [
            DownloadedTask(task_id=task, local_path=task_paths[task.name])
            for task in tasks
        ]

    def get_task_path(self, task: RegistryTaskId) -> Path | None:
        path = self._get_task_cache_path(task)
        return path if path.exists() else None
