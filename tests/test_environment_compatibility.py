from __future__ import annotations

from pathlib import Path

from sentient_evals.environments.base import EnvironmentType
from sentient_evals.environments.compatibility import collect_compatibility_issues
from sentient_evals.task_bundles.loader import load_task_bundle


def _write_task(root: Path, *, image: str | None = None, docker_from: str = "python:3.11-slim") -> Path:
    task_dir = root / "task1"
    env_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    env_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    image_line = f'image = "{image}"\n' if image is not None else ""
    (task_dir / "task.toml").write_text(
        (
            'id = "task1"\n'
            "[input]\n"
            'instruction = "solve"\n'
            "[environment]\n"
            'type = "container"\n'
            f"{image_line}"
        ),
        encoding="utf-8",
    )
    (task_dir / "instruction.md").write_text("do work", encoding="utf-8")
    (env_dir / "Dockerfile").write_text(f"FROM {docker_from}\nWORKDIR /workspace\n", encoding="utf-8")
    (tests_dir / "test.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    return task_dir


def test_e2b_incompatible_with_dockerfile_without_template_id(tmp_path: Path):
    task_dir = _write_task(tmp_path, image=None, docker_from="swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest")
    bundle = load_task_bundle(task_dir)
    issues = collect_compatibility_issues([bundle], EnvironmentType.e2b)
    assert len(issues) == 1
    assert "Dockerfile image build/runtime" in issues[0].reason
    assert "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest" in issues[0].reason


def test_e2b_accepts_template_id_in_environment_image(tmp_path: Path):
    task_dir = _write_task(tmp_path, image="base", docker_from="python:3.11-slim")
    bundle = load_task_bundle(task_dir)
    issues = collect_compatibility_issues([bundle], EnvironmentType.e2b)
    assert issues == []


def test_e2b_rejects_docker_image_reference_in_environment_image(tmp_path: Path):
    task_dir = _write_task(tmp_path, image="swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest")
    bundle = load_task_bundle(task_dir)
    issues = collect_compatibility_issues([bundle], EnvironmentType.e2b)
    assert len(issues) == 1
    assert "not an E2B template id" in issues[0].reason
