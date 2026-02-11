from __future__ import annotations

from pathlib import Path

from .base import EnvironmentConfig
from .cloud import CloudSandboxEnvironment, CloudSandboxSettings
from .providers import E2BProvider, SandboxCreateParams, SandboxResources


class E2BEnvironment(CloudSandboxEnvironment):
    def __init__(
        self,
        *,
        trial_id: str,
        workspace_dir: Path,
        logs_dir: Path,
        config: EnvironmentConfig,
        environment_dir: Path | None,
        image: str | None,
    ):
        dockerfile = None
        if environment_dir is not None:
            candidate = environment_dir / "Dockerfile"
            if candidate.exists():
                dockerfile = candidate

        resources = SandboxResources(
            cpus=config.cpus,
            memory_mb=config.memory_mb,
            storage_mb=config.storage_mb,
            gpus=config.gpus,
        )
        params = SandboxCreateParams(
            image=image,
            snapshot=None,
            dockerfile=dockerfile,
            context_dir=environment_dir,
            resources=resources,
            network_block_all=not config.allow_internet,
            build_timeout_sec=config.build_timeout_sec,
        )
        settings = CloudSandboxSettings(
            remote_workspace="/workspace",
            remote_logs="/logs",
            default_cwd="/workspace",
        )
        super().__init__(
            trial_id=trial_id,
            workspace_dir=workspace_dir,
            logs_dir=logs_dir,
            config=config,
            provider=E2BProvider(),
            create_params=params,
            settings=settings,
        )
