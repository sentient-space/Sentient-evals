from __future__ import annotations

from pathlib import Path

from sentient_evals.cli import _CLOUD_PROVIDER_API_KEYS
from sentient_evals.environments.base import EnvironmentConfig, EnvironmentType
from sentient_evals.environments.factory import EnvironmentFactory
from sentient_evals.prompts import _ENVIRONMENT_TYPES


def test_environment_type_includes_e2b():
    assert EnvironmentType.e2b.value == "e2b"


def test_prompt_envs_include_e2b():
    assert "e2b" in _ENVIRONMENT_TYPES


def test_cli_cloud_key_mapping_includes_e2b():
    assert _CLOUD_PROVIDER_API_KEYS["e2b"][0] == "E2B_API_KEY"


def test_factory_creates_e2b_environment(tmp_path: Path):
    env = EnvironmentFactory.create(
        env_type=EnvironmentType.e2b,
        trial_id="trial-1",
        workspace_dir=tmp_path / "workspace",
        logs_dir=tmp_path / "logs",
        cfg=EnvironmentConfig(),
        task_environment_dir=tmp_path / "task_environment",
        container_image=None,
    )
    assert env.__class__.__name__ == "E2BEnvironment"
    settings = getattr(env, "_settings")
    assert settings.remote_workspace == "/workspace"
    assert settings.remote_logs == "/logs"


def test_factory_creates_e2b_environment_with_testbed_workdir(tmp_path: Path):
    env_dir = tmp_path / "task_environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text("FROM python:3.11-slim\nWORKDIR /testbed\n", encoding="utf-8")

    env = EnvironmentFactory.create(
        env_type=EnvironmentType.e2b,
        trial_id="trial-1",
        workspace_dir=tmp_path / "workspace",
        logs_dir=tmp_path / "logs",
        cfg=EnvironmentConfig(),
        task_environment_dir=env_dir,
        container_image=None,
    )
    settings = getattr(env, "_settings")
    assert settings.remote_workspace == "/testbed"
    assert settings.default_cwd == "/testbed"


def test_factory_creates_e2b_environment_respects_workdir_override(tmp_path: Path):
    env_dir = tmp_path / "task_environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text("FROM python:3.11-slim\nWORKDIR /testbed\n", encoding="utf-8")

    env = EnvironmentFactory.create(
        env_type=EnvironmentType.e2b,
        trial_id="trial-1",
        workspace_dir=tmp_path / "workspace",
        logs_dir=tmp_path / "logs",
        cfg=EnvironmentConfig(),
        task_environment_dir=env_dir,
        container_image=None,
        container_workdir="/app",
    )
    settings = getattr(env, "_settings")
    assert settings.remote_workspace == "/app"
    assert settings.default_cwd == "/app"


def test_factory_e2b_includes_runtime_env_in_provider_options(tmp_path: Path):
    env = EnvironmentFactory.create(
        env_type=EnvironmentType.e2b,
        trial_id="trial-1",
        workspace_dir=tmp_path / "workspace",
        logs_dir=tmp_path / "logs",
        cfg=EnvironmentConfig(runtime_env={"COMPOSIO_API_KEY": "secret-key"}),
        task_environment_dir=tmp_path / "task_environment",
        container_image=None,
    )
    params = getattr(env, "_create_params")
    assert params.provider_options is not None
    assert params.provider_options["sandbox_env"] == {"COMPOSIO_API_KEY": "secret-key"}
