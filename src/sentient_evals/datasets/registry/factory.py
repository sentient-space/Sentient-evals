"""Factory for creating registry clients."""

from __future__ import annotations

from pathlib import Path

from .base import BaseRegistryClient
from .json import JsonRegistryClient


class RegistryClientFactory:
    """Factory for creating appropriate registry client instances."""

    @staticmethod
    def create(
        registry_url: str | None = None,
        registry_path: Path | None = None,
    ) -> BaseRegistryClient:
        """
        Create a registry client based on configuration.

        Args:
            registry_url: URL to a remote registry.json file
            registry_path: Path to a local registry.json file

        Returns:
            Configured registry client instance
        """
        if registry_url is not None and registry_path is not None:
            raise ValueError("Provide only one of registry_url or registry_path")

        if registry_path is not None:
            return JsonRegistryClient(path=registry_path)

        if registry_url is not None:
            return JsonRegistryClient(url=registry_url)

        raise ValueError(
            "No registry source specified. "
            "Provide --registry-url or --registry-path"
        )
