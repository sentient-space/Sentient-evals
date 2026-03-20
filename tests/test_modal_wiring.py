from __future__ import annotations

from pathlib import Path

from sentient_evals.cli import _CLOUD_PROVIDER_API_KEYS
from sentient_evals.environments.base import EnvironmentConfig, EnvironmentType
from sentient_evals.environments.factory import EnvironmentFactory
from sentient_evals.prompts import _ENVIRONMENT_TYPES


def test_environment_type_includes_modal():
    assert EnvironmentType.modal.value == "modal"


def test_prompt_envs_include_modal():
    assert "modal" in _ENVIRONMENT_TYPES


def test_cli_cloud_key_mapping_includes_modal():
    assert _CLOUD_PROVIDER_API_KEYS["modal"][0] == "MODAL_TOKEN_ID"


def test_factory_creates_modal_environment(tmp_path: Path):
    env = EnvironmentFactory.create(
        env_type=EnvironmentType.modal,
        trial_id="trial-1",
        workspace_dir=tmp_path / "workspace",
        logs_dir=tmp_path / "logs",
        cfg=EnvironmentConfig(),
        task_environment_dir=tmp_path / "task_environment",
        container_image=None,
    )
    assert env.__class__.__name__ == "ModalEnvironment"
    settings = getattr(env, "_settings")
    assert settings.remote_workspace == "/workspace"
    assert settings.remote_logs == "/logs"


def test_factory_preserves_modal_provider_options(tmp_path: Path):
    env = EnvironmentFactory.create(
        env_type=EnvironmentType.modal,
        trial_id="trial-1",
        workspace_dir=tmp_path / "workspace",
        logs_dir=tmp_path / "logs",
        cfg=EnvironmentConfig(),
        task_environment_dir=tmp_path / "task_environment",
        container_image="123456789012.dkr.ecr.us-east-1.amazonaws.com/private:latest",
        provider_options={
            "aws_ecr_secret_env": {
                "AWS_ACCESS_KEY_ID": "key",
                "AWS_SECRET_ACCESS_KEY": "secret",
                "AWS_REGION": "us-east-1",
            }
        },
    )

    params = getattr(env, "_create_params")
    assert params.provider_options is not None
    assert params.provider_options["aws_ecr_secret_env"] == {
        "AWS_ACCESS_KEY_ID": "key",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "us-east-1",
    }


def test_factory_creates_modal_environment_with_testbed_workdir(tmp_path: Path):
    env_dir = tmp_path / "task_environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text("FROM python:3.11-slim\nWORKDIR /testbed\n", encoding="utf-8")

    env = EnvironmentFactory.create(
        env_type=EnvironmentType.modal,
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


def test_factory_creates_modal_environment_respects_workdir_override(tmp_path: Path):
    env_dir = tmp_path / "task_environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text("FROM python:3.11-slim\nWORKDIR /testbed\n", encoding="utf-8")

    env = EnvironmentFactory.create(
        env_type=EnvironmentType.modal,
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


def test_factory_modal_includes_runtime_env_in_provider_options(tmp_path: Path):
    env = EnvironmentFactory.create(
        env_type=EnvironmentType.modal,
        trial_id="trial-1",
        workspace_dir=tmp_path / "workspace",
        logs_dir=tmp_path / "logs",
        cfg=EnvironmentConfig(runtime_env={"COMPOSIO_API_KEY": "secret-key"}),
        task_environment_dir=tmp_path / "task_environment",
        container_image=None,
    )
    params = getattr(env, "_create_params")
    assert params.provider_options is not None
    assert params.provider_options["sandbox_secret_env"] == {"COMPOSIO_API_KEY": "secret-key"}
