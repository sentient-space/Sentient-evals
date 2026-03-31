
from __future__ import annotations

import os
from pathlib import Path

from .base import BaseRegistryClient
from .github import DEFAULT_BRANCH, DEFAULT_GITHUB_REPO, GitHubRegistryClient
from .json import JsonRegistryClient


class RegistryClientFactory:

    @staticmethod
    def create(
        registry_url: str | None = None,
        registry_path: Path | None = None,
        use_default: bool = False,
        github_token: str | None = None,
    ) -> BaseRegistryClient:
        """
        Create a registry client based on configuration.

        Args:
            registry_url: URL to a remote registry.json file
            registry_path: Path to a local registry.json file
            use_default: If True and no other source provided, use default GitHub registry
            github_token: Optional GitHub token for higher rate limits (5000/hour vs 60/hour)

        Returns:
            Configured registry client instance
        """
        if registry_url is not None and registry_path is not None:
            raise ValueError("Provide only one of registry_url or registry_path")

        if registry_path is not None:
            return JsonRegistryClient(path=registry_path)

        if registry_url is not None:
            return JsonRegistryClient(url=registry_url)

        if use_default:
            token = github_token or os.environ.get("GITHUB_TOKEN")
            repo = os.environ.get("SENTIENT_EVALS_GITHUB_REPO") or None
            branch = os.environ.get("SENTIENT_EVALS_GITHUB_BRANCH") or None
            return GitHubRegistryClient(
                repo=repo or DEFAULT_GITHUB_REPO,
                branch=branch or DEFAULT_BRANCH,
                token=token,
            )

        raise ValueError(
            "No registry source specified. "
            "Provide --registry-url or --registry-path"
        )
