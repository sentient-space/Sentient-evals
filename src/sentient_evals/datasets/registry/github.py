
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any

import requests

from ..models import DatasetSpec, RegistryTaskId
from .base import BaseRegistryClient


# Default harbor-datasets repository
DEFAULT_GITHUB_REPO = "laude-institute/harbor-datasets"
DEFAULT_BRANCH = "main"
DATASETS_PATH = "datasets"

GITHUB_API_BASE = "https://api.github.com"
MAX_WORKERS = 10


class GitHubRegistryClient(BaseRegistryClient):

    def __init__(
        self,
        repo: str = DEFAULT_GITHUB_REPO,
        branch: str = DEFAULT_BRANCH,
        token: str | None = None,
    ):
        self._repo = repo
        self._branch = branch
        self._token = token
        self._session = requests.Session()
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"
        self._session.headers["Accept"] = "application/vnd.github.v3+json"
        self._session.headers["User-Agent"] = "sentient-evals"

    @property
    def _is_authenticated(self) -> bool:
        """Check if we have a GitHub token for higher rate limits."""
        return self._token is not None

    def _contents_url(self, path: str) -> str:
        """Build GitHub Contents API URL."""
        return f"{GITHUB_API_BASE}/repos/{self._repo}/contents/{path}?ref={self._branch}"

    def _fetch_json(self, url: str) -> list[dict[str, Any]]:
        """Fetch JSON from URL with error handling."""
        try:
            response = self._session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                raise ValueError(
                    "GitHub API rate limit exceeded. "
                    "Set GITHUB_TOKEN environment variable for higher limits (5000/hour)."
                ) from e
            if e.response.status_code == 404:
                raise ValueError(f"Path not found: {url}") from e
            raise
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to fetch from GitHub: {e}") from e

    @lru_cache(maxsize=1)
    def _fetch_datasets_list(self) -> list[dict[str, Any]]:
        """Fetch list of dataset directories (1 API call)."""
        url = self._contents_url(DATASETS_PATH)
        items = self._fetch_json(url)
        return [item for item in items if item.get("type") == "dir"]

    def _fetch_task_count(self, dataset_name: str) -> int:
        """Fetch task count for a single dataset (1 API call)."""
        url = self._contents_url(f"{DATASETS_PATH}/{dataset_name}")
        try:
            items = self._fetch_json(url)
            return sum(1 for item in items if item.get("type") == "dir")
        except ValueError:
            return -1  # Indicates error (rate limit, etc.)

    @lru_cache(maxsize=1)
    def _fetch_all_task_counts(self) -> dict[str, int]:
        """
        Fetch task counts for all datasets in parallel.
        
        Only called when authenticated (GITHUB_TOKEN set).
        Returns dict mapping dataset name to task count (-1 means unknown/error).
        """
        datasets = self._fetch_datasets_list()
        dataset_names = [item["name"] for item in datasets]
        
        task_counts: dict[str, int] = {}
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_name = {
                executor.submit(self._fetch_task_count, name): name
                for name in dataset_names
            }
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    task_counts[name] = future.result()
                except Exception:
                    task_counts[name] = -1
        
        return task_counts

    def get_datasets(self) -> list[DatasetSpec]:
        datasets_list = self._fetch_datasets_list()
        
        # Only fetch task counts if authenticated
        if self._is_authenticated:
            task_counts = self._fetch_all_task_counts()
        else:
            task_counts = {}  # Skip fetching, will show as unknown
        
        datasets = []
        for item in datasets_list:
            name = item["name"]
            count = task_counts.get(name, -1)
            task_refs = []
            if count > 0:
                task_refs = [
                    RegistryTaskId(
                        name=f"task_{i}",
                        git_url=f"https://github.com/{self._repo}.git",
                        git_commit_id=None,
                        path=f"{DATASETS_PATH}/{name}/task_{i}",
                    )
                    for i in range(count)
                ]
            
            spec = DatasetSpec(
                name=name,
                version="head",
                description=f"Dataset from {self._repo}",
                tasks=task_refs,
            )
            datasets.append(spec)
        
        return sorted(datasets, key=lambda d: d.name)

    def get_dataset_versions(self, name: str) -> list[str]:
        """Get available versions for a dataset (only 'head' for GitHub)."""
        datasets = self._fetch_datasets_list()
        names = {item["name"] for item in datasets}
        if name not in names:
            raise ValueError(f"Dataset '{name}' not found in {self._repo}")
        return ["head"]

    def _get_dataset_spec(self, name: str, version: str) -> DatasetSpec:
        """Get full dataset specification with actual task names."""
        if version != "head":
            raise ValueError(
                f"GitHub registry only supports version='head', got '{version}'"
            )
        
        datasets = self._fetch_datasets_list()
        if not any(item["name"] == name for item in datasets):
            raise ValueError(f"Dataset '{name}' not found in {self._repo}")
        
        # Fetch actual tasks for this dataset
        url = self._contents_url(f"{DATASETS_PATH}/{name}")
        contents = self._fetch_json(url)
        
        task_refs = [
            RegistryTaskId(
                name=item["name"],
                git_url=f"https://github.com/{self._repo}.git",
                git_commit_id=None,
                path=f"{DATASETS_PATH}/{name}/{item['name']}",
            )
            for item in contents
            if item.get("type") == "dir"
        ]
        
        return DatasetSpec(
            name=name,
            version="head",
            description=f"Dataset from {self._repo} ({len(task_refs)} tasks)",
            tasks=task_refs,
        )
