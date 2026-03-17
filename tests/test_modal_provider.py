from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from sentient_evals.environments.providers.base import SandboxCreateParams, SandboxResources
from sentient_evals.environments.providers.modal import ModalProvider


class _AioCallable:
    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    async def aio(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeStream:
    def __init__(self, text: str):
        self._text = text

    def read(self):
        return self._text


class _FakeProcess:
    def __init__(self, *, stdout: str, stderr: str, exit_code: int):
        self.stdout = _FakeStream(stdout)
        self.stderr = _FakeStream(stderr)
        self._exit_code = exit_code
        self.wait = _AioCallable(self._wait)

    def _wait(self):
        return self._exit_code


class _FakeFileHandle:
    def __init__(self, files: dict[str, bytes], path: str, mode: str):
        self._files = files
        self._path = path
        self._mode = mode
        self._read_pos = 0
        self._files.setdefault(path, b"")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def write(self, data: bytes):
        if "w" not in self._mode and "a" not in self._mode:
            raise ValueError("handle not writable")
        self._files[self._path] += bytes(data)
        return len(data)

    def read(self, size: int = -1):
        data = self._files.get(self._path, b"")
        if size is None or size < 0:
            start = self._read_pos
            self._read_pos = len(data)
            return data[start:]
        start = self._read_pos
        end = min(len(data), start + size)
        self._read_pos = end
        return data[start:end]


class _FakeSandbox:
    def __init__(self) -> None:
        self.id = "modal-sandbox-1"
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = {"/", "/workspace", "/logs"}
        self.exec_calls: list[dict[str, object]] = []
        self.terminate_calls = 0
        self.wait_calls = 0

        self.exec = _AioCallable(self._exec)
        self.mkdir = _AioCallable(self._mkdir)
        self.open = _AioCallable(self._open)
        self.ls = _AioCallable(self._ls)
        self.terminate = _AioCallable(self._terminate)
        self.wait = _AioCallable(self._wait)

    def _normalize_path(self, value: str) -> str:
        value = value.strip()
        if not value:
            return "/"
        if not value.startswith("/"):
            value = "/" + value
        if len(value) > 1 and value.endswith("/"):
            value = value.rstrip("/")
        return value

    def _mkdir(self, path: str, parents: bool = False):
        path = self._normalize_path(path)
        if parents:
            parts = path.strip("/").split("/")
            current = ""
            for part in parts:
                current += "/" + part
                self.dirs.add(current or "/")
        self.dirs.add(path)
        return True

    def _open(self, path: str, mode: str):
        path = self._normalize_path(path)
        if "w" in mode:
            self.files[path] = b""
        return _FakeFileHandle(self.files, path, mode)

    def _ls(self, path: str):
        path = self._normalize_path(path)
        if path in self.files:
            raise NotADirectoryError(path)
        if path not in self.dirs:
            raise FileNotFoundError(path)
        prefix = "/" if path == "/" else f"{path}/"
        children: set[str] = set()
        for directory in self.dirs:
            if not directory.startswith(prefix) or directory == path:
                continue
            rest = directory[len(prefix) :]
            if rest:
                children.add(rest.split("/", 1)[0])
        for file_path in self.files:
            if not file_path.startswith(prefix):
                continue
            rest = file_path[len(prefix) :]
            if rest:
                children.add(rest.split("/", 1)[0])
        return sorted(children)

    def _exec(self, *args, timeout=None):
        self.exec_calls.append({"args": args, "timeout": timeout})
        command = " ".join(str(a) for a in args)
        return _FakeProcess(stdout=f"ran:{command}", stderr="", exit_code=0)

    def _terminate(self):
        self.terminate_calls += 1
        return True

    def _wait(self, raise_on_termination: bool = False):
        self.wait_calls += 1
        return 0


def _build_fake_modal():
    state: dict[str, object] = {
        "app_lookup": None,
        "image_call": None,
        "create_kwargs": None,
    }

    class App:
        @staticmethod
        def _lookup(name: str, create_if_missing: bool = False):
            state["app_lookup"] = {"name": name, "create_if_missing": create_if_missing}
            return {"name": name}

    App.lookup = _AioCallable(App._lookup)

    class Image:
        @staticmethod
        def from_dockerfile(path: str, context_dir: str | None = None):
            state["image_call"] = ("dockerfile", path, context_dir)
            return ("dockerfile-image", path, context_dir)

        @staticmethod
        def from_registry(image: str):
            state["image_call"] = ("registry", image)
            return ("registry-image", image)

        @staticmethod
        def from_aws_ecr(image: str, secret):
            state["image_call"] = ("aws_ecr", image, secret)
            return ("aws-ecr-image", image, secret)

        @staticmethod
        def debian_slim():
            state["image_call"] = ("debian_slim",)
            return ("debian-image",)

    class Secret:
        @staticmethod
        def from_name(name: str):
            return f"secret:{name}"

        @staticmethod
        def from_dict(values: dict[str, str]):
            return ("secret-dict", dict(values))

    class Volume:
        @staticmethod
        def from_name(name: str, create_if_missing: bool = False):
            return f"volume:{name}:{create_if_missing}"

    class Sandbox:
        @staticmethod
        def _create(**kwargs):
            state["create_kwargs"] = kwargs
            return _FakeSandbox()

    Sandbox.create = _AioCallable(Sandbox._create)

    return SimpleNamespace(
        App=App,
        Image=Image,
        Secret=Secret,
        Volume=Volume,
        Sandbox=Sandbox,
        _state=state,
    )


def _provider_with_fake_modal(monkeypatch):
    fake_modal = _build_fake_modal()
    monkeypatch.setattr(ModalProvider, "_modal_sdk", staticmethod(lambda: fake_modal))
    return ModalProvider(), fake_modal


@pytest.mark.asyncio
async def test_modal_provider_create_with_dockerfile_and_options(monkeypatch, tmp_path: Path):
    provider, fake_modal = _provider_with_fake_modal(monkeypatch)
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.11-slim\n", encoding="utf-8")

    sandbox = await provider.create(
        SandboxCreateParams(
            dockerfile=dockerfile,
            context_dir=tmp_path,
            resources=SandboxResources(cpus=2, memory_mb=3072, gpus=1),
            network_block_all=True,
            provider_options={
                "app_name": "sentient-app",
                "secret_names": ("service-secret",),
                "volumes": {"/cache": "cache-vol"},
                "cidr_allowlist": ("10.0.0.0/8",),
            },
        )
    )

    assert isinstance(sandbox, _FakeSandbox)
    assert fake_modal._state["image_call"] == ("dockerfile", str(dockerfile), str(tmp_path))
    kwargs = fake_modal._state["create_kwargs"]
    assert kwargs is not None
    assert kwargs["cpu"] == 2
    assert kwargs["memory"] == 3072
    assert kwargs["gpu"] == "any:1"
    assert kwargs["block_network"] is True
    assert kwargs["cidr_allowlist"] == ["10.0.0.0/8"]
    assert kwargs["secrets"] == ["secret:service-secret"]
    assert kwargs["volumes"]["/cache"] == "volume:cache-vol:True"
    assert kwargs["timeout"] == 86_400


@pytest.mark.asyncio
async def test_modal_provider_image_selection(monkeypatch):
    provider, fake_modal = _provider_with_fake_modal(monkeypatch)

    await provider.create(SandboxCreateParams(image="python:3.11-slim"))
    assert fake_modal._state["image_call"] == ("registry", "python:3.11-slim")

    await provider.create(SandboxCreateParams())
    assert fake_modal._state["image_call"] == ("debian_slim",)


@pytest.mark.asyncio
async def test_modal_provider_uses_explicit_aws_ecr_secret_env(monkeypatch):
    provider, fake_modal = _provider_with_fake_modal(monkeypatch)

    await provider.create(
        SandboxCreateParams(
            image="123456789012.dkr.ecr.us-east-1.amazonaws.com/private:latest",
            provider_options={
                "aws_ecr_secret_env": {
                    "AWS_ACCESS_KEY_ID": "key",
                    "AWS_SECRET_ACCESS_KEY": "secret",
                    "AWS_REGION": "us-east-1",
                    "AWS_SESSION_TOKEN": "token",
                }
            },
        )
    )

    assert fake_modal._state["image_call"] == (
        "aws_ecr",
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/private:latest",
        (
            "secret-dict",
            {
                "AWS_ACCESS_KEY_ID": "key",
                "AWS_SECRET_ACCESS_KEY": "secret",
                "AWS_REGION": "us-east-1",
                "AWS_SESSION_TOKEN": "token",
            },
        ),
    )


@pytest.mark.asyncio
async def test_modal_provider_uses_named_aws_ecr_secret(monkeypatch):
    provider, fake_modal = _provider_with_fake_modal(monkeypatch)

    await provider.create(
        SandboxCreateParams(
            image="123456789012.dkr.ecr.us-east-1.amazonaws.com/private:latest",
            provider_options={"aws_ecr_secret_name": "aws-ecr"},
        )
    )

    assert fake_modal._state["image_call"] == (
        "aws_ecr",
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/private:latest",
        "secret:aws-ecr",
    )


@pytest.mark.asyncio
async def test_modal_provider_requires_explicit_ecr_auth(monkeypatch):
    provider, _fake_modal = _provider_with_fake_modal(monkeypatch)

    with pytest.raises(RuntimeError, match="aws_ecr_secret_env"):
        await provider.create(
            SandboxCreateParams(
                image="123456789012.dkr.ecr.us-east-1.amazonaws.com/private:latest",
            )
        )


@pytest.mark.asyncio
async def test_modal_provider_exec_wraps_cwd_env(monkeypatch):
    provider, _fake_modal = _provider_with_fake_modal(monkeypatch)
    sandbox = _FakeSandbox()

    result = await provider.exec(
        sandbox,
        "echo hello",
        cwd="/workspace",
        env={"API_KEY": "abc 123"},
        timeout_s=7,
    )

    assert result.exit_code == 0
    assert sandbox.exec_calls
    args = sandbox.exec_calls[-1]["args"]
    assert args[0] == "bash"
    assert args[1] == "-lc"
    wrapped = str(args[2])
    assert "cd /workspace" in wrapped
    assert "export API_KEY='abc 123'" in wrapped
    assert "echo hello" in wrapped


@pytest.mark.asyncio
async def test_modal_provider_upload_and_download_dir(monkeypatch, tmp_path: Path):
    provider, _fake_modal = _provider_with_fake_modal(monkeypatch)
    sandbox = _FakeSandbox()

    src = tmp_path / "src"
    (src / "nested").mkdir(parents=True)
    (src / "a.txt").write_text("alpha", encoding="utf-8")
    (src / "nested" / "b.txt").write_text("beta", encoding="utf-8")

    await provider.upload_dir(sandbox, src, "/workspace/input")
    assert sandbox.files["/workspace/input/a.txt"] == b"alpha"
    assert sandbox.files["/workspace/input/nested/b.txt"] == b"beta"

    out = tmp_path / "out"
    await provider.download_dir(sandbox, "/workspace/input", out)
    assert (out / "a.txt").read_text(encoding="utf-8") == "alpha"
    assert (out / "nested" / "b.txt").read_text(encoding="utf-8") == "beta"


@pytest.mark.asyncio
async def test_modal_provider_stop_and_delete(monkeypatch):
    provider, _fake_modal = _provider_with_fake_modal(monkeypatch)
    sandbox = _FakeSandbox()

    await provider.stop(sandbox)
    await provider.delete(sandbox)

    assert sandbox.terminate_calls == 2
    assert sandbox.wait_calls == 2
