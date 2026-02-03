"""JSON-based registry client for local files and remote URLs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

from ..models import DatasetSpec, Registry
from .base import BaseRegistryClient


class JsonRegistryClient(BaseRegistryClient):
    """Registry client that loads datasets from JSON (local file or remote URL)."""

    def __init__(self, url: str | None = None, path: Path | None = None):
        if url is not None and path is not None:
            raise ValueError("Provide only one of url or path, not both")
        if url is None and path is None:
            raise ValueError("Provide either url or path")

        self._url = url
        self._path = path
        self._registry: Registry | None = None

    def _load_registry(self) -> Registry:
        if self._registry is not None:
            return self._registry

        if self._url is not None:
            response = requests.get(self._url, timeout=30)
            response.raise_for_status()
            datasets_data = response.json()
        else:
            assert self._path is not None
            import json
            datasets_data = json.loads(self._path.read_text(encoding="utf-8"))

        if isinstance(datasets_data, list):
            datasets = [DatasetSpec.model_validate(d) for d in datasets_data]
            self._registry = Registry(
                url=self._url,
                path=self._path,
                datasets=datasets,
            )
        else:
            self._registry = Registry.model_validate(datasets_data)

        return self._registry

    @property
    def _datasets_by_name(self) -> dict[str, dict[str, DatasetSpec]]:
        registry = self._load_registry()
        result: dict[str, dict[str, DatasetSpec]] = defaultdict(dict)
        for ds in registry.datasets:
            result[ds.name][ds.version] = ds
        return result

    def get_datasets(self) -> list[DatasetSpec]:
        return self._load_registry().datasets

    def get_dataset_versions(self, name: str) -> list[str]:
        datasets = self._datasets_by_name
        if name not in datasets:
            raise ValueError(f"Dataset '{name}' not found")
        return list(datasets[name].keys())

    def _get_dataset_spec(self, name: str, version: str) -> DatasetSpec:
        datasets = self._datasets_by_name
        if name not in datasets:
            raise ValueError(f"Dataset '{name}' not found")
        if version not in datasets[name]:
            available = list(datasets[name].keys())
            raise ValueError(
                f"Version '{version}' not found for dataset '{name}'. "
                f"Available: {available}"
            )
        return datasets[name][version]
