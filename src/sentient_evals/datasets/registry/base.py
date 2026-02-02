

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from ..models import DatasetSpec, DownloadedTask


def resolve_version(versions: list[str]) -> str:
    """
    Resolve the best version from a list of versions.

    Priority:
    1. "head" if it exists
    2. Highest semantic version (PEP 440)
    3. Lexically last version
    """
    if not versions:
        raise ValueError("No versions available")

    if "head" in versions:
        return "head"

    semver_versions: list[tuple[Version, str]] = []
    for v in versions:
        try:
            semver_versions.append((Version(v), v))
        except InvalidVersion:
            pass

    if semver_versions:
        semver_versions.sort(key=lambda x: x[0], reverse=True)
        return semver_versions[0][1]

    return sorted(versions, reverse=True)[0]


class BaseRegistryClient(ABC):
    """Abstract base class for dataset registry clients."""

    @abstractmethod
    def get_datasets(self) -> list["DatasetSpec"]:
        """Get all datasets available in the registry."""
        ...

    @abstractmethod
    def get_dataset_versions(self, name: str) -> list[str]:
        """Get all available versions for a dataset."""
        ...

    @abstractmethod
    def _get_dataset_spec(self, name: str, version: str) -> "DatasetSpec":
        """Internal method to get a dataset spec by name and version."""
        ...

    def get_dataset_spec(
        self, name: str, version: str | None = None
    ) -> "DatasetSpec":
        """
        Get a dataset spec by name and optional version.

        If version is not specified, resolves the best version.
        """
        if version is None:
            versions = self.get_dataset_versions(name)
            version = resolve_version(versions)
        return self._get_dataset_spec(name, version)

    def download_dataset(
        self,
        name: str,
        version: str | None = None,
        output_dir: Path | None = None,
        overwrite: bool = False,
    ) -> list["DownloadedTask"]:
        """Download all tasks in a dataset to local cache."""
        from ..tasks_client import TaskClient

        spec = self.get_dataset_spec(name, version)
        task_client = TaskClient()
        return task_client.download_tasks(
            spec.tasks, output_dir=output_dir, overwrite=overwrite
        )
