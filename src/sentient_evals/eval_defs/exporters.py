from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .loader import discover_eval_definition_file, resolve_tasks_root
from .models import UnifiedEvalDefinition


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return '""'
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def definition_to_toml(definition: UnifiedEvalDefinition) -> str:
    lines: list[str] = [f'schema_version = "{definition.schema_version}"', ""]

    lines.extend(
        [
            "[metadata]",
            f"id = {_format_scalar(definition.metadata.id)}",
            f"name = {_format_scalar(definition.metadata.name)}",
            f'version = "{definition.metadata.version}"',
            f"description = {_format_scalar(definition.metadata.description)}",
            "labels = [" + ", ".join(_format_scalar(label) for label in definition.metadata.labels) + "]",
            "",
            "[dataset]",
            f'kind = "{definition.dataset.kind}"',
        ]
    )
    if definition.dataset.path:
        lines.append(f"path = {_format_scalar(definition.dataset.path)}")
    if definition.dataset.registry_ref:
        lines.append(f"registry_ref = {_format_scalar(definition.dataset.registry_ref)}")
    if definition.dataset.source:
        lines.append(f"source_json = {_format_scalar(json.dumps(definition.dataset.source, sort_keys=True))}")

    lines.extend(
        [
            "",
            "[execution]",
            "supported_run_targets = ["
            + ", ".join(_format_scalar(target) for target in definition.execution.supported_run_targets)
            + "]",
        ]
    )
    if definition.execution.env:
        lines.append(f"env = {_format_scalar(definition.execution.env)}")
    if definition.execution.suite_config:
        lines.append(
            f"suite_config_json = {_format_scalar(json.dumps(definition.execution.suite_config, sort_keys=True))}"
        )
    for evaluator in definition.execution.evaluators:
        if not isinstance(evaluator, dict):
            continue
        lines.extend(["", "[[execution.evaluators]]"])
        for key, value in evaluator.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{key}_json = {_format_scalar(json.dumps(value, sort_keys=True))}")
            else:
                lines.append(f"{key} = {_format_scalar(value)}")

    lines.extend(
        [
            "",
            "[provenance]",
            f'schema_version = "{definition.provenance.schema_version}"',
            f'generated_at = "{definition.provenance.generated_at.isoformat()}"',
        ]
    )
    if definition.provenance.harness_version:
        lines.append(f"harness_version = {_format_scalar(definition.provenance.harness_version)}")
    if definition.provenance.git_sha:
        lines.append(f"git_sha = {_format_scalar(definition.provenance.git_sha)}")
    if definition.provenance.connector_snapshot:
        lines.append(
            "connector_snapshot_json = "
            + _format_scalar(json.dumps(definition.provenance.connector_snapshot, sort_keys=True))
        )

    for task in definition.embedded_tasks:
        lines.extend(["", "[[embedded_tasks]]"])
        lines.append(f"id = {_format_scalar(task.id)}")
        lines.append(f"name = {_format_scalar(task.name)}")
        lines.append(f"instruction = {_format_scalar(task.instruction)}")
        if task.expected_output is not None:
            lines.append(f"expected_output = {_format_scalar(task.expected_output)}")
        if task.timeout_seconds is not None:
            lines.append(f"timeout_seconds = {_format_scalar(task.timeout_seconds)}")
        if task.test_script is not None:
            lines.append(f"test_script = {_format_scalar(task.test_script)}")
        lines.append(
            "tags = [" + ", ".join(_format_scalar(tag) for tag in task.tags) + "]"
        )
        if task.task_config:
            lines.append(f"task_config_json = {_format_scalar(json.dumps(task.task_config, sort_keys=True))}")
        if task.environment_spec:
            lines.append(
                "environment_spec_json = "
                + _format_scalar(json.dumps(task.environment_spec, sort_keys=True))
            )
        if task.metadata:
            lines.append(f"metadata_json = {_format_scalar(json.dumps(task.metadata, sort_keys=True))}")

    if definition.sentient:
        lines.extend(["", "[sentient]"])
        for key, value in definition.sentient.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{key}_json = {_format_scalar(json.dumps(value, sort_keys=True))}")
            else:
                lines.append(f"{key} = {_format_scalar(value)}")

    return "\n".join(lines) + "\n"


def write_definition_export(
    definition: UnifiedEvalDefinition,
    *,
    source_path: Path,
    output_path: Path,
    export_format: str,
) -> Path:
    export_format = export_format.lower()
    output_path = output_path.resolve()

    if export_format == "json":
        output_path.write_text(
            json.dumps(definition.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    if export_format == "toml":
        output_path.write_text(definition_to_toml(definition), encoding="utf-8")
        return output_path

    if export_format != "bundle":
        raise ValueError(f"Unsupported export format: {export_format}")

    output_path.mkdir(parents=True, exist_ok=True)
    tasks_root = resolve_tasks_root(source_path)
    bundle_tasks_dir = output_path / "tasks"
    if bundle_tasks_dir.exists():
        shutil.rmtree(bundle_tasks_dir)
    shutil.copytree(tasks_root, bundle_tasks_dir)
    (output_path / "eval.toml").write_text(definition_to_toml(definition), encoding="utf-8")

    source_definition = discover_eval_definition_file(source_path)
    if source_definition and source_definition.suffix.lower() == ".json":
        (output_path / "eval.json").write_text(source_definition.read_text(encoding="utf-8"), encoding="utf-8")
    return output_path
