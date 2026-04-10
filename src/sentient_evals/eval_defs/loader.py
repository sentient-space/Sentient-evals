from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from sentient_evals import __version__
from sentient_evals.task_bundles import load_task_bundles

from .models import (
    EmbeddedTaskDefinition,
    EvalDefinitionDataset,
    EvalDefinitionExecution,
    EvalDefinitionMetadata,
    EvalDefinitionProvenance,
    UnifiedEvalDefinition,
)


def discover_eval_definition_file(path: Path) -> Path | None:
    candidate = path if path.is_file() else path / "eval.toml"
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def resolve_tasks_root(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        return path.parent
    if (path / "task.toml").exists():
        return path
    tasks_dir = path / "tasks"
    if tasks_dir.exists() and tasks_dir.is_dir():
        return tasks_dir
    return path


def _load_raw_eval_definition(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return _normalize_loaded_definition(raw)


def _normalize_loaded_definition(raw: dict[str, Any]) -> dict[str, Any]:
    dataset = raw.get("dataset")
    if isinstance(dataset, dict) and isinstance(dataset.get("source_json"), str):
        dataset["source"] = json.loads(dataset.pop("source_json"))

    execution = raw.get("execution")
    if isinstance(execution, dict):
        if isinstance(execution.get("suite_config_json"), str):
            execution["suite_config"] = json.loads(execution.pop("suite_config_json"))
        evaluators = execution.get("evaluators")
        if isinstance(evaluators, list):
            for spec in evaluators:
                if isinstance(spec, dict) and isinstance(spec.get("config_json"), str):
                    spec["config"] = json.loads(spec.pop("config_json"))

    provenance = raw.get("provenance")
    if isinstance(provenance, dict) and isinstance(provenance.get("connector_snapshot_json"), str):
        provenance["connector_snapshot"] = json.loads(provenance.pop("connector_snapshot_json"))
    return raw


def load_eval_definition(path: Path) -> UnifiedEvalDefinition:
    definition_file = discover_eval_definition_file(path)
    if definition_file is None:
        raise FileNotFoundError(f"No eval definition found under {path}")
    raw = _load_raw_eval_definition(definition_file)
    return UnifiedEvalDefinition.model_validate(raw)


def build_definition_from_task_bundles(
    tasks_root: Path,
    *,
    evaluators: list[dict[str, Any]] | None = None,
    suite_config: dict[str, Any] | None = None,
    env: str | None = None,
) -> UnifiedEvalDefinition:
    bundles = load_task_bundles(tasks_root)
    metadata = EvalDefinitionMetadata(
        id=tasks_root.name,
        name=tasks_root.name,
        description="Generated from Harbor-style task bundles",
    )
    dataset = EvalDefinitionDataset(kind="task_bundles", path=str(tasks_root))
    execution = EvalDefinitionExecution(
        evaluators=list(evaluators or []),
        suite_config=dict(suite_config or {}),
        env=env,
        supported_run_targets=["local"],
    )
    provenance = EvalDefinitionProvenance(harness_version=__version__)
    embedded_tasks = [
        EmbeddedTaskDefinition(
            id=bundle.task.id,
            name=bundle.task.id,
            instruction=bundle.instruction,
            task_config=dict(bundle.task.input or {}),
            test_script=(
                (bundle.tests_dir / "test.sh").read_text(encoding="utf-8")
                if bundle.tests_dir and (bundle.tests_dir / "test.sh").exists()
                else None
            ),
            environment_spec=bundle.env.model_dump(mode="json"),
            metadata=dict(bundle.task.metadata or {}),
            timeout_seconds=bundle.task.timeout_seconds,
        )
        for bundle in bundles
    ]
    return UnifiedEvalDefinition(
        metadata=metadata,
        dataset=dataset,
        embedded_tasks=embedded_tasks,
        execution=execution,
        provenance=provenance,
    )


def load_or_build_definition(
    path: Path,
    *,
    evaluators: list[dict[str, Any]] | None = None,
    suite_config: dict[str, Any] | None = None,
    env: str | None = None,
) -> UnifiedEvalDefinition:
    definition_file = discover_eval_definition_file(path)
    if definition_file is not None:
        definition = load_eval_definition(definition_file)
        if definition.dataset.kind == "task_bundles" and not definition.embedded_tasks:
            if definition.dataset.path:
                task_root = Path(definition.dataset.path)
                if not task_root.is_absolute():
                    task_root = (definition_file.parent / task_root).resolve()
            else:
                task_root = resolve_tasks_root(path)
            generated = build_definition_from_task_bundles(
                task_root,
                evaluators=definition.execution.evaluators or evaluators,
                suite_config=definition.execution.suite_config or suite_config,
                env=definition.execution.env or env,
            )
            definition.embedded_tasks = generated.embedded_tasks
            if definition.metadata.id is None:
                definition.metadata.id = generated.metadata.id
            if definition.metadata.name is None:
                definition.metadata.name = generated.metadata.name
        return definition
    return build_definition_from_task_bundles(
        resolve_tasks_root(path),
        evaluators=evaluators,
        suite_config=suite_config,
        env=env,
    )
