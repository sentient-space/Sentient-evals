from __future__ import annotations

from pathlib import Path

from .loader import resolve_tasks_root
from .models import UnifiedEvalDefinition


def validate_definition(definition: UnifiedEvalDefinition, *, base_dir: Path | None = None) -> list[str]:
    issues: list[str] = []

    if definition.dataset.kind == "task_bundles":
        task_root = None
        if definition.dataset.path:
            task_root = Path(definition.dataset.path)
            if base_dir is not None and not task_root.is_absolute():
                task_root = (base_dir / task_root).resolve()
        elif base_dir is not None:
            task_root = resolve_tasks_root(base_dir)

        if task_root is None or not task_root.exists():
            issues.append("Task bundle dataset path does not exist")
        else:
            tasks_root = resolve_tasks_root(task_root)
            task_dirs = [p for p in tasks_root.iterdir() if p.is_dir()] if tasks_root.is_dir() else []
            has_single_task = (tasks_root / "task.toml").exists()
            if not has_single_task and not any((p / "task.toml").exists() for p in task_dirs):
                issues.append("No task bundles found under the provided path")

    requires_verifier = any(
        str(spec.get("type") or "").strip() == "verifier_script"
        for spec in definition.execution.evaluators
        if isinstance(spec, dict)
    )
    if requires_verifier:
        if definition.dataset.kind == "task_bundles" and base_dir is not None:
            tasks_root = resolve_tasks_root(base_dir)
            missing = []
            if (tasks_root / "task.toml").exists():
                candidates = [tasks_root]
            else:
                candidates = [p for p in tasks_root.iterdir() if p.is_dir() and (p / "task.toml").exists()]
            for candidate in candidates:
                if not (candidate / "tests" / "test.sh").exists():
                    missing.append(candidate.name)
            if missing:
                issues.append(f"Verifier evaluator requires tests/test.sh for task bundles: {', '.join(missing)}")
        elif definition.embedded_tasks:
            missing = [task.id for task in definition.embedded_tasks if not (task.test_script or "").strip()]
            if missing:
                issues.append(f"Verifier evaluator requires test_script for embedded tasks: {', '.join(missing)}")

    return issues
