from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import BaseEnvironment, EnvironmentConfig, EnvironmentType


@dataclass(frozen=True)
class EnvironmentFactoryConfig:
    env_type: EnvironmentType
    docker_image_tag_prefix: str = "sentient-evals"
    daytona_snapshot_template: str | None = None
    daytona_network_block_all: bool | None = None
    modal_app_name: str | None = None
    modal_secrets: tuple[str, ...] = ()
    modal_volumes: tuple[str, ...] = ()
    modal_cidr_allowlist: tuple[str, ...] = ()
    modal_allow_network: bool = False
    extra: dict[str, Any] | None = None


class EnvironmentFactory:
    @staticmethod
    def create(
        *,
        env_type: EnvironmentType,
        trial_id: str,
        workspace_dir: Path,
        logs_dir: Path,
        cfg: EnvironmentConfig,
        task_environment_dir: Path | None = None,
        task_files_dir: Path | None = None,
        task_digest: str | None = None,
        container_image: str | None = None,
        container_platform: str | None = None,
        container_workdir: str | None = None,
        container_workspace_mount: str | None = None,
        docker_image_tag_prefix: str = "sentient-evals",
        daytona_snapshot_template: str | None = None,
        daytona_network_block_all: bool | None = None,
        modal_app_name: str | None = None,
        modal_secrets: tuple[str, ...] = (),
        modal_volumes: tuple[str, ...] = (),
        modal_cidr_allowlist: tuple[str, ...] = (),
        modal_allow_network: bool = False,
        adapter_name: str | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> BaseEnvironment:
        if env_type == EnvironmentType.local_python:
            from .local_python import LocalPythonEnvironment

            return LocalPythonEnvironment(
                trial_id=trial_id,
                workspace_dir=workspace_dir,
                logs_dir=logs_dir,
                config=cfg,
                task_files_dir=task_files_dir,
            )

        if env_type in (EnvironmentType.docker_cli, EnvironmentType.podman_cli):
            from .docker_cli import DockerCLIEnvironment

            return DockerCLIEnvironment(
                trial_id=trial_id,
                workspace_dir=workspace_dir,
                logs_dir=logs_dir,
                config=cfg,
                engine="docker" if env_type == EnvironmentType.docker_cli else "podman",
                environment_dir=task_environment_dir,
                task_digest=task_digest,
                image_tag_prefix=docker_image_tag_prefix,
                platform=container_platform,
                workdir=container_workdir,
                workspace_mount=container_workspace_mount,
            )

        if env_type == EnvironmentType.docker_sdk:
            from .docker_sdk import DockerSDKEnvironment

            return DockerSDKEnvironment(
                trial_id=trial_id,
                workspace_dir=workspace_dir,
                logs_dir=logs_dir,
                config=cfg,
                environment_dir=task_environment_dir,
                task_digest=task_digest,
                image_tag_prefix=docker_image_tag_prefix,
                platform=container_platform,
                workdir=container_workdir,
                workspace_mount=container_workspace_mount,
            )

        if env_type == EnvironmentType.daytona:
            from .daytona import DaytonaEnvironment

            return DaytonaEnvironment(
                trial_id=trial_id,
                workspace_dir=workspace_dir,
                logs_dir=logs_dir,
                config=cfg,
                environment_dir=task_environment_dir,
                task_digest=task_digest,
                image=container_image,
                snapshot_template_name=daytona_snapshot_template,
                network_block_all=daytona_network_block_all,
                workdir=container_workdir,
                provider_options=provider_options,
            )

        if env_type == EnvironmentType.e2b:
            from .e2b import E2BEnvironment

            return E2BEnvironment(
                trial_id=trial_id,
                workspace_dir=workspace_dir,
                logs_dir=logs_dir,
                config=cfg,
                environment_dir=task_environment_dir,
                image=container_image,
                adapter_name=adapter_name,
                workdir=container_workdir,
                provider_options=provider_options,
            )

        if env_type == EnvironmentType.modal:
            from .modal import ModalEnvironment

            return ModalEnvironment(
                trial_id=trial_id,
                workspace_dir=workspace_dir,
                logs_dir=logs_dir,
                config=cfg,
                environment_dir=task_environment_dir,
                image=container_image,
                app_name=modal_app_name,
                secret_names=modal_secrets,
                volume_specs=modal_volumes,
                cidr_allowlist=modal_cidr_allowlist,
                allow_network_override=modal_allow_network,
                workdir=container_workdir,
                provider_options=provider_options,
            )

        raise ValueError(f"Unsupported environment type: {env_type}")
