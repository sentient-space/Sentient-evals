from __future__ import annotations

from pathlib import Path

from .base import EnvironmentConfig
from .cloud import CloudSandboxEnvironment, CloudSandboxSettings
from .providers import DaytonaProvider, SandboxCreateParams, SandboxResources


class DaytonaEnvironment(CloudSandboxEnvironment):
    def __init__(
        self,
        *,
        trial_id: str,
        workspace_dir: Path,
        logs_dir: Path,
        config: EnvironmentConfig,
        environment_dir: Path | None,
        task_digest: str | None,
        image: str | None,
        snapshot_template_name: str | None = None,
        network_block_all: bool | None = None,
    ):
        dockerfile = None
        if environment_dir is not None:
            candidate = environment_dir / "Dockerfile"
            if candidate.exists():
                dockerfile = candidate

        snapshot_name = None
        if snapshot_template_name and task_digest:
            snapshot_name = snapshot_template_name.format(name=task_digest[:12])

        if network_block_all is None:
            network_block_all = not config.allow_internet

        resources = SandboxResources(
            cpus=config.cpus,
            memory_mb=config.memory_mb,
            storage_mb=config.storage_mb,
            gpus=config.gpus,
        )
        params = SandboxCreateParams(
            image=image,
            snapshot=snapshot_name,
            dockerfile=dockerfile,
            context_dir=environment_dir,
            resources=resources,
            network_block_all=network_block_all,
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
            provider=DaytonaProvider(),
            create_params=params,
            settings=settings,
        )
