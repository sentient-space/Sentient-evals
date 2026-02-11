from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from sentient_evals.environments.providers.base import SandboxCreateParams
from sentient_evals.environments.providers.e2b import E2BProvider


@dataclass
class _CmdResult:
    stdout: str
    stderr: str
    exit_code: int


class _FakeFiles:
    def __init__(self) -> None:
        self.storage: dict[str, bytes] = {}

    def write(self, path: str, *, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        self.storage[path] = bytes(data)
        return {"ok": True}

    def read(self, path: str, format: str = "bytes"):
        value = self.storage.get(path, b"")
        if format == "bytes":
            return value
        return value.decode("utf-8", errors="replace")


class _FakeCommands:
    def __init__(self, files: _FakeFiles) -> None:
        self.files = files
        self.calls: list[dict[str, object]] = []

    def run(
        self,
        cmd: str,
        *,
        cwd: str | None = None,
        envs: dict[str, str] | None = None,
        timeout: float | None = None,
        request_timeout: float | None = None,
        background: bool | None = None,
    ):
        self.calls.append(
            {
                "cmd": cmd,
                "cwd": cwd,
                "envs": envs,
                "timeout": timeout,
                "request_timeout": request_timeout,
                "background": background,
            }
        )
        if "find . -type f -print" in cmd:
            src = cmd.split("cd ", 1)[1].split(" && ", 1)[0].strip("'")
            files = sorted(
                p[len(src) + 1 :] for p in self.files.storage if p.startswith(f"{src}/")
            )
            return _CmdResult(stdout="\n".join(files), stderr="", exit_code=0)
        return _CmdResult(stdout=f"ran:{cmd}", stderr="", exit_code=0)


class _FakeSandbox:
    def __init__(self) -> None:
        self.id = "sbx-test-1"
        self.files = _FakeFiles()
        self.commands = _FakeCommands(self.files)
        self.killed = False

    def kill(self):
        self.killed = True
        return True


class _FakeSandboxClass:
    last_kwargs: dict[str, object] | None = None

    @classmethod
    def beta_create(cls, **kwargs):
        cls.last_kwargs = kwargs
        return _FakeSandbox()


@pytest.mark.asyncio
async def test_e2b_provider_create_and_exec(monkeypatch):
    monkeypatch.setattr(E2BProvider, "_sandbox_cls", staticmethod(lambda: _FakeSandboxClass))
    monkeypatch.setenv("E2B_API_KEY", "Bearer   test-key-123  ")
    provider = E2BProvider()
    sandbox = await provider.create(
        SandboxCreateParams(image="template-123", network_block_all=True, build_timeout_sec=90)
    )

    assert isinstance(sandbox, _FakeSandbox)
    assert _FakeSandboxClass.last_kwargs is not None
    assert _FakeSandboxClass.last_kwargs["template"] == "template-123"
    assert _FakeSandboxClass.last_kwargs["allow_internet_access"] is False
    assert _FakeSandboxClass.last_kwargs["api_key"] == "test-key-123"
    assert _FakeSandboxClass.last_kwargs["timeout"] == 3_600

    result = await provider.exec(sandbox, "echo hello", cwd="/workspace", timeout_s=5)
    assert result.exit_code == 0
    assert "echo hello" in result.stdout


@pytest.mark.asyncio
async def test_e2b_provider_uses_explicit_base_template(monkeypatch):
    monkeypatch.setattr(E2BProvider, "_sandbox_cls", staticmethod(lambda: _FakeSandboxClass))
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    provider = E2BProvider()
    sandbox = await provider.create(SandboxCreateParams())
    assert isinstance(sandbox, _FakeSandbox)
    assert _FakeSandboxClass.last_kwargs is not None
    assert _FakeSandboxClass.last_kwargs["template"] == "base"


@pytest.mark.asyncio
async def test_e2b_provider_upload_and_download_dir(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(E2BProvider, "_sandbox_cls", staticmethod(lambda: _FakeSandboxClass))
    provider = E2BProvider()
    sandbox = _FakeSandbox()

    src = tmp_path / "src"
    (src / "nested").mkdir(parents=True)
    (src / "a.txt").write_text("alpha", encoding="utf-8")
    (src / "nested" / "b.txt").write_text("beta", encoding="utf-8")

    await provider.upload_dir(sandbox, src, "/workspace/input")
    assert sandbox.files.storage["/workspace/input/a.txt"] == b"alpha"
    assert sandbox.files.storage["/workspace/input/nested/b.txt"] == b"beta"

    out = tmp_path / "out"
    await provider.download_dir(sandbox, "/workspace/input", out)
    assert (out / "a.txt").read_text(encoding="utf-8") == "alpha"
    assert (out / "nested" / "b.txt").read_text(encoding="utf-8") == "beta"


@pytest.mark.asyncio
async def test_e2b_provider_stop_and_delete(monkeypatch):
    monkeypatch.setattr(E2BProvider, "_sandbox_cls", staticmethod(lambda: _FakeSandboxClass))
    provider = E2BProvider()
    sandbox = _FakeSandbox()

    await provider.stop(sandbox)
    assert sandbox.killed is True
    sandbox.killed = False

    await provider.delete(sandbox)
    assert sandbox.killed is True
