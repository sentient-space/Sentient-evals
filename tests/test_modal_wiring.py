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
