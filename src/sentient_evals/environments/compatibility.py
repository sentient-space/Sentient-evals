from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .base import EnvironmentType
from ..task_bundles.models import TaskBundle


@dataclass(frozen=True)
class CompatibilityIssue:
    task_id: str
    reason: str


_FROM_RE = re.compile(r"^\s*FROM\s+([^\s#]+)", re.IGNORECASE | re.MULTILINE)


def _read_first_from(dockerfile: Path | None) -> str | None:
    if dockerfile is None or not dockerfile.exists():
        return None
    content = dockerfile.read_text(encoding="utf-8")
    match = _FROM_RE.search(content)
    if not match:
        return None
    return match.group(1).strip()


def _is_e2b_template_id(value: str) -> bool:
    candidate = value.strip()
    if not candidate:
        return False
    # Docker image references almost always include "/" and frequently ":".
    if "/" in candidate or ":" in candidate or "@" in candidate:
        return False
    return True


def validate_bundle_compatibility(bundle: TaskBundle, env_type: EnvironmentType) -> CompatibilityIssue | None:
    if env_type != EnvironmentType.e2b:
        return None

    bundle_env_type = getattr(bundle.env, "type", "")
    if bundle_env_type != "container":
        return None

    configured_image = getattr(bundle.env, "image", None)
    if isinstance(configured_image, str) and configured_image.strip():
        if _is_e2b_template_id(configured_image):
            return None
        return CompatibilityIssue(
            task_id=bundle.task.id,
            reason=(
                "environment.image is not an E2B template id. "
                f"Got '{configured_image}'. For E2B use a template id like 'base'."
            ),
        )

    dockerfile = (bundle.environment_dir / "Dockerfile") if bundle.environment_dir else None
    from_image = _read_first_from(dockerfile)
    if from_image:
        return CompatibilityIssue(
            task_id=bundle.task.id,
            reason=(
                "task relies on Dockerfile image build/runtime, but E2B backend does not build Dockerfiles at run time. "
                f"Dockerfile FROM is '{from_image}'. "
                "Use docker/daytona for this dataset, or set [environment].image to a compatible E2B template id."
            ),
        )

    return CompatibilityIssue(
        task_id=bundle.task.id,
        reason=(
            "container task has no E2B template id configured. "
            "Set [environment].image to an E2B template id (for example, 'base')."
        ),
    )


def collect_compatibility_issues(
    bundles: Sequence[TaskBundle], env_type: EnvironmentType
) -> list[CompatibilityIssue]:
    issues: list[CompatibilityIssue] = []
    for bundle in bundles:
        issue = validate_bundle_compatibility(bundle, env_type)
        if issue is not None:
            issues.append(issue)
    return issues
