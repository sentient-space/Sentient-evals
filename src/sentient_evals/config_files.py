from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tomllib
from pydantic import BaseModel, Field

from .models import SuiteConfig


class AdapterSpecFile(BaseModel):
    type: str
    import_path: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)


class RunSpecFile(BaseModel):
    suite: SuiteConfig | None = None
    adapter: AdapterSpecFile
    graders: list[dict[str, Any]] = Field(default_factory=list)


def load_run_spec(path: Path) -> RunSpecFile:
    raw: dict[str, Any]
    if path.suffix.lower() == ".toml":
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
    return RunSpecFile.model_validate(raw)
