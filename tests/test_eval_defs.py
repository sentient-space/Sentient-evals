import json
from pathlib import Path

from typer.testing import CliRunner

from sentient_evals.cli import app
from sentient_evals.eval_defs import definition_to_toml, load_eval_definition, load_or_build_definition, validate_definition
from sentient_evals.task_bundles import load_task_bundles


runner = CliRunner()


def _write_task(task_dir: Path, *, task_id: str = "t1", with_test: bool = True) -> None:
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("do thing", encoding="utf-8")
    (task_dir / "task.toml").write_text(
        '\n'.join(
            [
                f'id = "{task_id}"',
                "timeout_seconds = 30",
                "",
                "[input]",
                'question = "?"',
                "",
                "[metadata]",
                'difficulty = "easy"',
                "",
                "[environment]",
                'type = "container"',
                "allow_internet = true",
            ]
        ),
        encoding="utf-8",
    )
    if with_test:
        (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")


def test_load_task_bundles_supports_dataset_root(tmp_path: Path):
    dataset_root = tmp_path / "demo"
    task_dir = dataset_root / "tasks" / "sample"
    _write_task(task_dir)

    bundles = load_task_bundles(dataset_root)
    assert len(bundles) == 1
    assert bundles[0].task.id == "t1"


def test_load_or_build_definition_from_dataset_root(tmp_path: Path):
    dataset_root = tmp_path / "demo"
    task_dir = dataset_root / "tasks" / "sample"
    _write_task(task_dir)

    definition = load_or_build_definition(dataset_root)
    assert definition.dataset.kind == "task_bundles"
    assert definition.dataset.path == str((dataset_root / "tasks").resolve())
    assert definition.embedded_tasks[0].id == "t1"


def test_validate_definition_flags_missing_verifier_tests(tmp_path: Path):
    dataset_root = tmp_path / "demo"
    task_dir = dataset_root / "tasks" / "sample"
    _write_task(task_dir, with_test=False)

    definition = load_or_build_definition(dataset_root)
    definition.execution.evaluators = [{"type": "verifier_script", "config": {}}]
    issues = validate_definition(definition, base_dir=dataset_root)
    assert issues
    assert "tests/test.sh" in issues[0]


def test_cli_init_scaffolds_dataset_root(tmp_path: Path):
    result = runner.invoke(app, ["init", "demo-eval", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    root = tmp_path / "demo-eval"
    assert (root / "eval.toml").exists()
    assert (root / "tasks" / "starter_task" / "task.toml").exists()


def test_cli_tasks_create_adds_task_under_dataset_root(tmp_path: Path):
    root = tmp_path / "demo"
    runner.invoke(app, ["init", "demo", "--output-dir", str(tmp_path)])

    result = runner.invoke(app, ["tasks", "create", "follow_up", "--output-dir", str(root)])
    assert result.exit_code == 0, result.stdout
    assert (root / "tasks" / "follow_up" / "task.toml").exists()


def test_cli_evals_validate_success(tmp_path: Path):
    root = tmp_path / "demo"
    runner.invoke(app, ["init", "demo", "--output-dir", str(tmp_path)])

    result = runner.invoke(app, ["evals", "validate", str(root)])
    assert result.exit_code == 0, result.stdout
    assert "valid" in result.stdout.lower()


def test_cli_evals_export_json(tmp_path: Path):
    root = tmp_path / "demo"
    runner.invoke(app, ["init", "demo", "--output-dir", str(tmp_path)])
    out = tmp_path / "definition.json"

    result = runner.invoke(app, ["evals", "export", str(root), "--format", "json", "--out", str(out)])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["dataset"]["kind"] == "task_bundles"
    assert payload["embedded_tasks"][0]["id"] == "starter_task"


def test_definition_toml_roundtrip_preserves_embedded_tasks_and_sentient_metadata(tmp_path: Path):
    root = tmp_path / "demo"
    task_dir = root / "tasks" / "sample"
    _write_task(task_dir)

    definition = load_or_build_definition(root)
    definition.sentient = {
        "source_suite_id": "suite-123",
        "criteria_templates": [{"id": "crit-1", "name": "Correctness"}],
    }
    definition.embedded_tasks[0].expected_output = "42"
    definition.embedded_tasks[0].metadata["origin"] = "generated"

    out = tmp_path / "eval.toml"
    out.write_text(definition_to_toml(definition), encoding="utf-8")

    loaded = load_eval_definition(out)
    assert loaded.sentient["source_suite_id"] == "suite-123"
    assert loaded.sentient["criteria_templates"][0]["id"] == "crit-1"
    assert loaded.embedded_tasks[0].expected_output == "42"
    assert loaded.embedded_tasks[0].metadata["origin"] == "generated"
