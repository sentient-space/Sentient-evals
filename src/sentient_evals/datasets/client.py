
from __future__ import annotations

from pathlib import Path

from .models import DatasetSpec, DownloadedTask
from .registry import BaseRegistryClient, RegistryClientFactory
from .tasks_client import TaskClient


class DatasetClient:
    """High-level API for dataset operations."""

    def __init__(
        self,
        registry_url: str | None = None,
        registry_path: Path | None = None,
    ):
        self._registry = RegistryClientFactory.create(
            registry_url=registry_url,
            registry_path=registry_path,
        )
        self._task_client = TaskClient()

    @property
    def registry(self) -> BaseRegistryClient:
        return self._registry

    def list_datasets(self) -> list[DatasetSpec]:
        return self._registry.get_datasets()

    def get_dataset(self, name: str, version: str | None = None) -> DatasetSpec:
        return self._registry.get_dataset_spec(name, version)

    def download_dataset(
        self,
        name: str,
        version: str | None = None,
        output_dir: Path | None = None,
        overwrite: bool = False,
    ) -> list[DownloadedTask]:
        spec = self.get_dataset(name, version)
        return self._task_client.download_tasks(
            spec.tasks,
            output_dir=output_dir,
            overwrite=overwrite,
        )

    def get_dataset_path(self, name: str, version: str | None = None) -> Path | None:
        spec = self.get_dataset(name, version)
        if not spec.tasks:
            return None
        return self._task_client.get_task_path(spec.tasks[0])
