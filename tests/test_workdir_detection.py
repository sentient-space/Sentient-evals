from __future__ import annotations

from sentient_evals.environments.workdir import detect_dockerfile_workdir, resolve_workdir


def test_detect_dockerfile_workdir_defaults_when_missing(tmp_path):
    assert detect_dockerfile_workdir(tmp_path / "missing") == "/workspace"


def test_detect_dockerfile_workdir_uses_last_workdir(tmp_path):
    env_dir = tmp_path / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text(
        "FROM python:3.11-slim\nWORKDIR /workspace\nWORKDIR /testbed\n",
        encoding="utf-8",
    )
    assert detect_dockerfile_workdir(env_dir) == "/testbed"


def test_detect_dockerfile_workdir_ignores_unresolved_env_var(tmp_path):
    env_dir = tmp_path / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text(
        "FROM python:3.11-slim\nWORKDIR $APP_HOME\n",
        encoding="utf-8",
    )
    assert detect_dockerfile_workdir(env_dir) == "/workspace"


def test_resolve_workdir_prefers_override(tmp_path):
    env_dir = tmp_path / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text(
        "FROM python:3.11-slim\nWORKDIR /testbed\n",
        encoding="utf-8",
    )
    assert resolve_workdir(env_dir, override="/app") == "/app"
