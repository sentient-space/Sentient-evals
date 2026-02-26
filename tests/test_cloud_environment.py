from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentient_evals.env import ExecResult
from sentient_evals.environments.cloud import CloudSandboxEnvironment
from sentient_evals.environments.base import EnvironmentConfig
from sentient_evals.environments.providers.base import (
    CloudSandboxProvider,
    SandboxCapabilities,
    SandboxCreateParams,
    SandboxResources,
)


class _FakeSandbox:
    def __init__(self, sandbox_id: str = "sandbox-123") -> None:
        self.id = sandbox_id


class _FakeProvider(CloudSandboxProvider):
    name = "fake"
    capabilities = SandboxCapabilities()

    def __init__(self, *, fail_first_create: bool = False) -> None:
        self.fail_first_create = fail_first_create
        self.create_calls = 0
        self.exec_calls = 0

    async def create(self, params: SandboxCreateParams):
        self.create_calls += 1
        if self.fail_first_create and self.create_calls == 1:
            raise RuntimeError("create failed")
        return _FakeSandbox()

    async def delete(self, sandbox):
        return None

    async def stop(self, sandbox):
        return None

    async def exec(self, sandbox, command: str, *, cwd=None, env=None, timeout_s=None):
        self.exec_calls += 1
        return ExecResult(stdout="ok", stderr="", exit_code=0, duration_ms=5)

    async def upload_file(self, sandbox, source_path: Path, target_path: str):
        return None

    async def upload_dir(self, sandbox, source_dir: Path, target_dir: str):
        return None

    async def download_file(self, sandbox, source_path: str, target_path: Path):
        return None

    async def download_dir(self, sandbox, source_dir: str, target_dir: Path):
        return None


@pytest.mark.asyncio
async def test_cloud_env_writes_env_info_and_events(tmp_path: Path):
    provider = _FakeProvider()
    cfg = EnvironmentConfig()
    params = SandboxCreateParams(resources=SandboxResources())
    env = CloudSandboxEnvironment(
        trial_id="t1",
        workspace_dir=tmp_path / "ws",
        logs_dir=tmp_path / "logs",
        config=cfg,
        provider=provider,
        create_params=params,
    )

    await env.start()
    await env.exec("echo ok")
    await env.stop()

    info_path = tmp_path / "logs" / "env_info.json"
    assert info_path.exists()
    info = json.loads(info_path.read_text())
    assert info["provider"] == "fake"
    assert info["sandbox_id"] == "sandbox-123"

    events_path = tmp_path / "logs" / "env_events.jsonl"
    assert events_path.exists()
    lines = events_path.read_text().splitlines()
    events = [json.loads(l) for l in lines]
    assert any(e.get("event") == "create_sandbox" for e in events)
    assert any(e.get("event") == "exec" for e in events)


@pytest.mark.asyncio
async def test_cloud_env_retries_create(tmp_path: Path):
    provider = _FakeProvider(fail_first_create=True)
    cfg = EnvironmentConfig()
    params = SandboxCreateParams(resources=SandboxResources())
    env = CloudSandboxEnvironment(
        trial_id="t2",
        workspace_dir=tmp_path / "ws",
        logs_dir=tmp_path / "logs",
        config=cfg,
        provider=provider,
        create_params=params,
    )

    await env.start()
    await env.stop()

    events_path = tmp_path / "logs" / "env_events.jsonl"
    lines = events_path.read_text().splitlines()
    create_events = [json.loads(l) for l in lines if json.loads(l).get("event") == "create_sandbox"]
    assert len(create_events) >= 2
    assert any(e.get("status") == "error" for e in create_events)
    assert any(e.get("status") == "ok" for e in create_events)
