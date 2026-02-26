from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .base import EnvironmentType
from ..task_bundles.models import TaskBundle


@dataclass(frozen=True)
class CompatibilityIssue:
    task_id: str
    reason: str


def validate_bundle_compatibility(bundle: TaskBundle, env_type: EnvironmentType) -> CompatibilityIssue | None:
    if env_type != EnvironmentType.e2b:
        return None

    bundle_env_type = getattr(bundle.env, "type", "")
    if bundle_env_type != "container":
        return None

    configured_image = getattr(bundle.env, "image", None)
    if isinstance(configured_image, str) and configured_image.strip():
        return None

    dockerfile = (bundle.environment_dir / "Dockerfile") if bundle.environment_dir else None
    if dockerfile is not None and dockerfile.exists():
        return None

    return CompatibilityIssue(
        task_id=bundle.task.id,
        reason=(
            "container task has no Dockerfile and no [environment].image configured for E2B. "
            "Provide an image (template id or Docker image reference) or add environment/Dockerfile."
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
