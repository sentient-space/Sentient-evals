from __future__ import annotations

from pathlib import Path

from .loader import load_or_build_definition, resolve_tasks_root
from .models import UnifiedEvalDefinition


def build_definition_for_path(
    path: Path,
    *,
    evaluators: list[dict] | None = None,
    suite_config: dict | None = None,
    env: str | None = None,
) -> UnifiedEvalDefinition:
    return load_or_build_definition(
        path,
        evaluators=evaluators,
        suite_config=suite_config,
        env=env,
    )


def resolve_tasks_dir_for_run(path: Path) -> Path:
    return resolve_tasks_root(path)
