from __future__ import annotations

import re
from pathlib import Path


_DEFAULT_WORKDIR = "/workspace"
_WORKDIR_PATTERN = re.compile(r"^\s*WORKDIR\s+(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _normalize_workdir(raw: str) -> str | None:
    value = raw.split("#", 1)[0].strip()
    if not value:
        return None
    if value.startswith("["):
        return None
    if (value[0] == value[-1]) and value[0] in {"'", '"'} and len(value) >= 2:
        value = value[1:-1].strip()
    if not value or "$" in value:
        return None
    if not value.startswith("/"):
        value = f"/{value}"
    return value.rstrip("/") or "/"


def detect_dockerfile_workdir(environment_dir: Path | None, *, default: str = _DEFAULT_WORKDIR) -> str:
    if environment_dir is None:
        return default

    dockerfile = environment_dir / "Dockerfile"
    if not dockerfile.exists():
        return default

    try:
        content = dockerfile.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = dockerfile.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return default

    matches = _WORKDIR_PATTERN.findall(content)
    for raw_match in reversed(matches):
        normalized = _normalize_workdir(raw_match)
        if normalized:
            return normalized

    return default


def resolve_workdir(
    environment_dir: Path | None,
    *,
    override: str | None = None,
    default: str = _DEFAULT_WORKDIR,
) -> str:
    normalized_override = _normalize_workdir(override or "")
    if normalized_override:
        return normalized_override
    return detect_dockerfile_workdir(environment_dir, default=default)
