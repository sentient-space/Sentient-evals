from __future__ import annotations

from pathlib import Path

from .base import EnvironmentConfig
from .cloud import CloudSandboxEnvironment, CloudSandboxSettings
from .providers import ModalProvider, SandboxCreateParams, SandboxResources


def _parse_modal_volume_specs(specs: tuple[str, ...]) -> dict[str, str]:
    volumes: dict[str, str] = {}
    for spec in specs:
        mount_path, sep, volume_name = spec.partition("=")
        mount_path = mount_path.strip()
        volume_name = volume_name.strip()
        if sep != "=" or not mount_path or not volume_name:
            raise ValueError(
                f"Invalid --modal-volume value '{spec}'. Expected format: mount_path=volume_name"
            )
        if not mount_path.startswith("/"):
            raise ValueError(
                f"Invalid --modal-volume mount path '{mount_path}'. Mount paths must be absolute."
            )
        if mount_path in volumes and volumes[mount_path] != volume_name:
            raise ValueError(
                f"Conflicting --modal-volume values for mount path '{mount_path}'."
            )
        volumes[mount_path] = volume_name
    return volumes


class ModalEnvironment(CloudSandboxEnvironment):
    def __init__(
        self,
        *,
        trial_id: str,
        workspace_dir: Path,
        logs_dir: Path,
        config: EnvironmentConfig,
        environment_dir: Path | None,
        image: str | None,
        app_name: str | None = None,
        secret_names: tuple[str, ...] = (),
        volume_specs: tuple[str, ...] = (),
        cidr_allowlist: tuple[str, ...] = (),
        allow_network_override: bool = False,
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

        block_network = not config.allow_internet
        if allow_network_override:
            block_network = False

        params = SandboxCreateParams(
            image=image,
            snapshot=None,
            dockerfile=dockerfile,
            context_dir=environment_dir,
            resources=resources,
            provider_options={
                "app_name": app_name,
                "secret_names": tuple(secret_names),
                "volumes": _parse_modal_volume_specs(tuple(volume_specs)),
                "cidr_allowlist": tuple(cidr_allowlist),
            },
            network_block_all=block_network,
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
            provider=ModalProvider(),
            create_params=params,
            settings=settings,
        )
