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
