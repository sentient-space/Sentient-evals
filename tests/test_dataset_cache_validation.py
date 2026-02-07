from __future__ import annotations

from pathlib import Path
import sys

import typer

from sentient_evals.cli import _resolve_dataset_tasks_dir
from sentient_evals.datasets.models import DatasetSpec, DownloadedTask, RegistryTaskId
from sentient_evals.datasets.registry.base import BaseRegistryClient


class FakeRegistryClient(BaseRegistryClient):
    def __init__(self, spec: DatasetSpec, *, download_tasks: list[str] | None = None) -> None:
        self._spec = spec
        # If provided, only these task dir basenames will be created during download.
        self._download_tasks = download_tasks
        self.download_calls = 0

    def get_datasets(self) -> list[DatasetSpec]:
        return [self._spec]

    def get_dataset_versions(self, name: str) -> list[str]:
        return [self._spec.version]

    def _get_dataset_spec(self, name: str, version: str) -> DatasetSpec:
        return self._spec

    def download_dataset(
        self,
        name: str,
        version: str | None = None,
        output_dir: Path | None = None,
        overwrite: bool = False,
    ) -> list[DownloadedTask]:
        self.download_calls += 1
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        created: list[DownloadedTask] = []
        allowed = set(self._download_tasks) if self._download_tasks is not None else None
        for task in self._spec.tasks:
            task_dirname = Path(task.path).name
            if allowed is not None and task_dirname not in allowed:
                continue
            task_dir = output_dir / task_dirname
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "task.toml").write_text(f'id = "{task_dirname}"\n', encoding="utf-8")
            created.append(DownloadedTask(task_id=task, local_path=task_dir))
        return created


def _make_spec() -> DatasetSpec:
    tasks = [
        RegistryTaskId(name="t1", git_url="https://example.com/repo.git", path="datasets/foo/task_1"),
        RegistryTaskId(name="t2", git_url="https://example.com/repo.git", path="datasets/foo/task_2"),
    ]
    return DatasetSpec(name="foo", version="head", tasks=tasks)


def test_resolve_dataset_tasks_dir_reuses_valid_cache(tmp_path, monkeypatch):
    spec = _make_spec()
    client = FakeRegistryClient(spec)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    cache_dir = tmp_path / ".cache" / "sentient-evals" / "datasets" / "foo@head"
    (cache_dir / "task_1").mkdir(parents=True)
    (cache_dir / "task_1" / "task.toml").write_text('id = "task_1"\n', encoding="utf-8")
    (cache_dir / "task_2").mkdir(parents=True)
    (cache_dir / "task_2" / "task.toml").write_text('id = "task_2"\n', encoding="utf-8")

    resolved = _resolve_dataset_tasks_dir(name="foo", version="head", client=client)
    assert resolved == cache_dir
    assert client.download_calls == 0


def test_resolve_dataset_tasks_dir_redownloads_incomplete_cache(tmp_path, monkeypatch):
    spec = _make_spec()
    client = FakeRegistryClient(spec)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    cache_dir = tmp_path / ".cache" / "sentient-evals" / "datasets" / "foo@head"
    (cache_dir / "task_1").mkdir(parents=True)
    (cache_dir / "task_1" / "task.toml").write_text('id = "task_1"\n', encoding="utf-8")

    resolved = _resolve_dataset_tasks_dir(name="foo", version="head", client=client)
    assert resolved == cache_dir
    assert (cache_dir / "task_1" / "task.toml").exists()
    assert (cache_dir / "task_2" / "task.toml").exists()
    assert client.download_calls == 1


def test_resolve_dataset_tasks_dir_raises_on_incomplete_download(tmp_path, monkeypatch):
    spec = _make_spec()
    # Only download one task to simulate an incomplete download.
    client = FakeRegistryClient(spec, download_tasks=["task_1"])

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    try:
        _resolve_dataset_tasks_dir(name="foo", version="head", client=client)
    except typer.BadParameter as exc:
        assert "incomplete" in str(exc).lower()
    else:
        raise AssertionError("Expected BadParameter due to incomplete download")
