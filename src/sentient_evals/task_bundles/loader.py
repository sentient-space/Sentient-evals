from __future__ import annotations

import hashlib
import os
import tomllib
from pathlib import Path
from typing import Any, Iterable

from sentient_evals.models import Task

from .models import (
    ContainerEnvironmentSpec,
    EnvironmentResources,
    EnvironmentSpec,
    LocalPythonEnvironmentSpec,
    TaskBundle,
)


_IGNORED_DIRS = {".git", "__pycache__", ".venv", ".mypy_cache", ".ruff_cache"}
_IGNORED_FILES = {".DS_Store"}


def _iter_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        dirnames[:] = sorted([x for x in dirnames if x not in _IGNORED_DIRS and not x.startswith(".")])
        for fn in sorted(filenames):
            if fn in _IGNORED_FILES or fn.startswith("."):
                continue
            p = d / fn
            if p.is_symlink():
                continue
            if p.is_file():
                yield p


def compute_task_bundle_digest(task_dir: Path) -> str:
    """
    Deterministic hash over the task directory contents for reproducibility.
    """
    h = hashlib.sha256()
    for p in _iter_files(task_dir):
        rel = p.relative_to(task_dir).as_posix().encode("utf-8")
        h.update(rel)
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _parse_env(cfg: dict[str, Any]) -> EnvironmentSpec:
    env_cfg = dict(cfg.get("environment") or {})
    env_type = str(env_cfg.get("type") or "local_python").strip()
    allow_internet = bool(env_cfg.get("allow_internet", True))
    build_timeout_sec = env_cfg.get("build_timeout_sec")
    platform = env_cfg.get("platform")
    workdir = env_cfg.get("workdir")
    workspace_mount = env_cfg.get("workspace_mount")

    res_cfg = dict(env_cfg.get("resources") or {})
    resources = EnvironmentResources(
        cpus=res_cfg.get("cpus"),
        memory_mb=res_cfg.get("memory_mb"),
        storage_mb=res_cfg.get("storage_mb"),
        gpus=res_cfg.get("gpus"),
    )

    if env_type == "container":
        return ContainerEnvironmentSpec(
            allow_internet=allow_internet,
            build_timeout_sec=build_timeout_sec,
            resources=resources,
            image=env_cfg.get("image"),
            platform=platform,
            workdir=workdir,
            workspace_mount=workspace_mount,
        )

    lp_cfg = dict(cfg.get("local_python") or {})
    return LocalPythonEnvironmentSpec(
        allow_internet=allow_internet,
        build_timeout_sec=build_timeout_sec,
        resources=resources,
        entrypoint=lp_cfg.get("entrypoint"),
    )


def load_task_bundle(task_dir: Path) -> TaskBundle:
    task_dir = task_dir.resolve()
    task_toml = task_dir / "task.toml"
    if not task_toml.exists():
        raise FileNotFoundError(task_toml)

    cfg = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    task_id = str(cfg.get("id") or task_dir.name)

    instruction_path = task_dir / "instruction.md"
    instruction = instruction_path.read_text(encoding="utf-8") if instruction_path.exists() else ""

    input_payload = dict(cfg.get("input") or {})
    metadata = dict(cfg.get("metadata") or {})

    timeout_seconds = cfg.get("timeout_seconds")
    task = Task(id=task_id, input=input_payload, metadata=metadata, timeout_seconds=timeout_seconds)

    env = _parse_env(cfg)
    digest = compute_task_bundle_digest(task_dir)

    env_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    files_dir = task_dir / "files"

    extra = {k: v for k, v in cfg.items() if k not in {"id", "input", "metadata", "timeout_seconds", "environment", "local_python"}}
    return TaskBundle(
        task=task,
        instruction=instruction,
        task_dir=task_dir,
        environment_dir=env_dir if env_dir.exists() else None,
        tests_dir=tests_dir if tests_dir.exists() else None,
        files_dir=files_dir if files_dir.exists() else None,
        env=env,
        digest=digest,
        extra=extra,
    )


def load_task_bundles(tasks_dir: Path) -> list[TaskBundle]:
    """
    Loads task bundles from either:
    - a single task directory containing task.toml, or
    - a directory of task directories (depth=1) where each child has task.toml.
    """
    tasks_dir = tasks_dir.resolve()
    if (tasks_dir / "task.toml").exists():
        return [load_task_bundle(tasks_dir)]

    bundles: list[TaskBundle] = []
    for child in sorted([p for p in tasks_dir.iterdir() if p.is_dir()]):
        if (child / "task.toml").exists():
            bundles.append(load_task_bundle(child))
    return bundles
