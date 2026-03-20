from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import EnvironmentConfig
from .cloud import CloudSandboxEnvironment, CloudSandboxSettings
from .providers import DaytonaProvider, SandboxCreateParams, SandboxResources
from .workdir import resolve_workdir


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
        workdir: str | None = None,
        provider_options: dict[str, Any] | None = None,
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
        merged_provider_options = dict(provider_options or {})
        if config.runtime_env:
            merged_provider_options.setdefault("sandbox_env", dict(config.runtime_env))

        params = SandboxCreateParams(
            image=image,
            snapshot=snapshot_name,
            dockerfile=dockerfile,
            context_dir=environment_dir,
            resources=resources,
            provider_options=merged_provider_options or None,
            network_block_all=network_block_all,
            build_timeout_sec=config.build_timeout_sec,
        )
        effective_workdir = resolve_workdir(
            environment_dir,
            override=workdir,
            default="/workspace",
        )

        settings = CloudSandboxSettings(
            remote_workspace=effective_workdir,
            remote_logs="/logs",
            default_cwd=effective_workdir,
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
