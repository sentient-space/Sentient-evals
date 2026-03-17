from __future__ import annotations

import asyncio
import logging
import shlex
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from sentient_evals.env import ExecResult

from .base import CloudSandboxProvider, SandboxCapabilities, SandboxCreateParams


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, bytearray):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, list):
        return "".join(_to_text(v) for v in value)
    if isinstance(value, tuple):
        return "".join(_to_text(v) for v in value)
    return str(value)


class ModalProvider(CloudSandboxProvider):
    name = "modal"
    _default_timeout_sec = 86_400
    _default_app_name = "__sentient_evals__"
    capabilities = SandboxCapabilities(
        supports_network_toggle=True,
        supports_snapshots=False,
        supports_gpus=True,
        supports_attach=False,
        supports_stop=True,
        max_upload_batch=None,
    )

    async def create(self, params: SandboxCreateParams) -> Any:
        modal = self._modal_sdk()
        options = params.provider_options or {}

        app_name = str(options.get("app_name") or self._default_app_name)
        secret_names = tuple(str(v) for v in options.get("secret_names", ()))
        cidr_allowlist = tuple(str(v) for v in options.get("cidr_allowlist", ()))
        volumes = options.get("volumes", {})
        if not isinstance(volumes, dict):
            raise ValueError("Modal provider option 'volumes' must be a dict of mount_path -> volume_name")

        app = await self._call_modal(modal.App.lookup, app_name, create_if_missing=True)
        image = self._resolve_image(modal, params)
        timeout = int(params.build_timeout_sec) if params.build_timeout_sec else self._default_timeout_sec

        create_kwargs: dict[str, Any] = {
            "app": app,
            "image": image,
            "timeout": max(1, timeout),
            "verbose": True, 
        }

        if params.resources is not None:
            if params.resources.cpus is not None:
                create_kwargs["cpu"] = params.resources.cpus
            if params.resources.memory_mb is not None:
                create_kwargs["memory"] = int(params.resources.memory_mb)
            if params.resources.gpus is not None and params.resources.gpus > 0:
                create_kwargs["gpu"] = f"any:{int(params.resources.gpus)}"

        if params.network_block_all is not None:
            create_kwargs["block_network"] = bool(params.network_block_all)
        if cidr_allowlist:
            create_kwargs["cidr_allowlist"] = list(cidr_allowlist)
        if secret_names:
            create_kwargs["secrets"] = [modal.Secret.from_name(name) for name in secret_names]
        if volumes:
            create_kwargs["volumes"] = {
                str(mount): modal.Volume.from_name(str(vol_name), create_if_missing=True)
                for mount, vol_name in volumes.items()
            }

        return await self._call_modal(modal.Sandbox.create, **create_kwargs)

    async def delete(self, sandbox: Any) -> None:
        await self._terminate(sandbox)

    async def stop(self, sandbox: Any) -> None:
        await self._terminate(sandbox)

    async def exec(
        self,
        sandbox: Any,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        wrapped = self._wrap_command(command, cwd=cwd, env=env)
        started = time.monotonic()
        process = await self._invoke_method(
            sandbox,
            "exec",
            "bash",
            "-lc",
            wrapped,
            timeout=int(timeout_s) if timeout_s is not None else None,
        )

       
        stdout_task = asyncio.create_task(
            self._read_stream_live(getattr(process, "stdout", ""), "stdout")
        )
        stderr_task = asyncio.create_task(
            self._read_stream_live(getattr(process, "stderr", ""), "stderr")
        )
        stdout, stderr = await asyncio.gather(stdout_task, stderr_task)

        exit_code = await self._wait_process(process)
        dur_ms = int((time.monotonic() - started) * 1000)
        return ExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code, duration_ms=dur_ms)

    async def upload_file(self, sandbox: Any, source_path: Path, target_path: str) -> None:
        await self._mkdir_parent(sandbox, target_path)
        async with self._sandbox_open(sandbox, target_path, "wb") as remote_file:
            with source_path.open("rb") as local_file:
                while True:
                    chunk = local_file.read(8192)
                    if not chunk:
                        break
                    await self._write_handle(remote_file, chunk)

    async def upload_dir(self, sandbox: Any, source_dir: Path, target_dir: str) -> None:
        await self._invoke_method(sandbox, "mkdir", target_dir, parents=True)
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file() and not file_path.is_symlink():
                rel = file_path.relative_to(source_dir).as_posix()
                await self.upload_file(sandbox, file_path, f"{target_dir.rstrip('/')}/{rel}")

    async def download_file(self, sandbox: Any, source_path: str, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._sandbox_open(sandbox, source_path, "rb") as remote_file:
            chunks: list[bytes] = []
            while True:
                chunk = await self._read_handle_chunk(remote_file, 8192)
                if not chunk:
                    break
                if isinstance(chunk, str):
                    chunks.append(chunk.encode("utf-8"))
                elif isinstance(chunk, bytearray):
                    chunks.append(bytes(chunk))
                elif isinstance(chunk, bytes):
                    chunks.append(chunk)
                else:
                    chunks.append(_to_text(chunk).encode("utf-8"))
            target_path.write_bytes(b"".join(chunks))

    async def download_dir(self, sandbox: Any, source_dir: str, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        children = await self._invoke_method(sandbox, "ls", source_dir)
        for child in children:
            child_name = self._child_name(child)
            if not child_name:
                continue
            remote_path = f"{source_dir.rstrip('/')}/{child_name}"
            local_path = target_dir / child_name
            if await self._is_directory(sandbox, remote_path):
                await self.download_dir(sandbox, remote_path, local_path)
            else:
                await self.download_file(sandbox, remote_path, local_path)

    def _resolve_image(self, modal: Any, params: SandboxCreateParams) -> Any:
        options = params.provider_options or {}
        if params.dockerfile is not None:
            if params.context_dir is not None:
                return modal.Image.from_dockerfile(
                    str(params.dockerfile),
                    context_dir=str(params.context_dir),
                )
            return modal.Image.from_dockerfile(str(params.dockerfile))
        if params.image:
            ecr_secret = self._resolve_aws_ecr_secret(modal, str(params.image), options)
            if ecr_secret is not None:
                return modal.Image.from_aws_ecr(str(params.image), secret=ecr_secret)
            return modal.Image.from_registry(params.image)
        return modal.Image.debian_slim()

    @staticmethod
    def _resolve_aws_ecr_secret(modal: Any, image: str, options: dict[str, Any]) -> Any | None:
        if ".dkr.ecr." not in image or ".amazonaws.com/" not in image:
            return None

        secret_name = options.get("aws_ecr_secret_name")
        if isinstance(secret_name, str) and secret_name.strip():
            return modal.Secret.from_name(secret_name.strip())

        raw_secret_env = options.get("aws_ecr_secret_env")
        if raw_secret_env is None:
            raise RuntimeError(
                "Modal private ECR images require provider_options['aws_ecr_secret_env'] "
                "or provider_options['aws_ecr_secret_name']."
            )
        if not isinstance(raw_secret_env, dict):
            raise ValueError("Modal provider option 'aws_ecr_secret_env' must be a dict of AWS credential env vars")

        secret_env = {
            str(key): str(value).strip()
            for key, value in raw_secret_env.items()
            if str(key).strip() and value is not None and str(value).strip()
        }
        required = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")
        missing = [key for key in required if not secret_env.get(key)]
        if missing:
            raise RuntimeError(
                "Modal private ECR auth is missing required AWS keys: " + ", ".join(missing)
            )
        return modal.Secret.from_dict(secret_env)

    @staticmethod
    def _modal_sdk():
        try:
            import modal  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError(
                "Modal SDK not installed. Install with `pip install \"sentient-evals[modal]\"`."
            ) from exc
        return modal

    @staticmethod
    async def _call_modal(fn, *args, **kwargs):
        aio = getattr(fn, "aio", None)
        if callable(aio):
            return await aio(*args, **kwargs)
        if asyncio.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        result = await asyncio.to_thread(fn, *args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _invoke_method(self, obj: Any, name: str, *args, **kwargs):
        method = getattr(obj, name, None)
        if not callable(method):
            raise RuntimeError(f"Modal sandbox does not expose `{name}`")
        return await self._call_modal(method, *args, **kwargs)

    async def _terminate(self, sandbox: Any) -> None:
        try:
            await self._invoke_method(sandbox, "terminate")
        finally:
            try:
                await self._invoke_method(sandbox, "wait", raise_on_termination=False)
            except Exception:
                pass

    @staticmethod
    def _wrap_command(command: str, *, cwd: str | None, env: dict[str, str] | None) -> str:
        parts: list[str] = []
        if cwd:
            parts.append(f"cd {shlex.quote(cwd)}")
        if env:
            for key, value in env.items():
                parts.append(f"export {key}={shlex.quote(value)}")
        parts.append(command)
        return " && ".join(parts)

    @asynccontextmanager
    async def _sandbox_open(self, sandbox: Any, path: str, mode: str):
        opener = getattr(sandbox, "open", None)
        if not callable(opener):
            raise RuntimeError("Modal sandbox does not expose `open`")
        handle = await self._call_modal(opener, path, mode)
        if hasattr(handle, "__aenter__"):
            async with handle as opened:
                yield opened
            return
        if hasattr(handle, "__enter__"):
            with handle as opened:
                yield opened
            return
        try:
            yield handle
        finally:
            close = getattr(handle, "close", None)
            if callable(close):
                await self._call_modal(close)

    async def _mkdir_parent(self, sandbox: Any, target_path: str) -> None:
        parent = str(Path(target_path).parent)
        await self._invoke_method(sandbox, "mkdir", parent, parents=True)

    async def _write_handle(self, handle: Any, data: bytes) -> None:
        write = getattr(handle, "write", None)
        if not callable(write):
            raise RuntimeError("Modal file handle does not expose `write`")
        await self._call_modal(write, data)

    async def _read_handle_chunk(self, handle: Any, size: int) -> bytes | str:
        read = getattr(handle, "read", None)
        if not callable(read):
            raise RuntimeError("Modal file handle does not expose `read`")
        try:
            return await self._call_modal(read, size)
        except TypeError:
            return await self._call_modal(read)

    async def _read_stream_live(self, stream: Any, label: str = "output") -> str:
        
        if stream is None:
            return ""

        
        if hasattr(stream, "__aiter__"):
            chunks: list[str] = []
            async for chunk in stream:
                text = _to_text(chunk)
                chunks.append(text)
                for line in text.splitlines():
                    if line.strip():
                        logger.info("[modal %s] %s", label, line)
            return "".join(chunks)

 
        return await self._read_stream(stream)

    async def _read_stream(self, stream: Any) -> str:
        if stream is None:
            return ""
        read = getattr(stream, "read", None)
        if callable(read):
            return _to_text(await self._call_modal(read))
        if hasattr(stream, "__aiter__"):
            chunks: list[str] = []
            async for chunk in stream:
                chunks.append(_to_text(chunk))
            return "".join(chunks)
        if hasattr(stream, "__iter__") and not isinstance(stream, (str, bytes, bytearray)):
            return "".join(_to_text(chunk) for chunk in stream)
        return _to_text(stream)

    async def _wait_process(self, process: Any) -> int:
        wait = getattr(process, "wait", None)
        if callable(wait):
            result = await self._call_modal(wait)
            try:
                return int(result)
            except Exception:
                pass
        for attr in ("exit_code", "return_code", "code"):
            value = getattr(process, attr, None)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    pass
        return 0

    async def _is_directory(self, sandbox: Any, path: str) -> bool:
        try:
            await self._invoke_method(sandbox, "ls", path)
            return True
        except Exception:
            return False

    @staticmethod
    def _child_name(entry: Any) -> str:
        if isinstance(entry, str):
            return entry.strip("/")
        for attr in ("name", "path"):
            value = getattr(entry, attr, None)
            if value:
                return str(value).rstrip("/").split("/")[-1]
        return _to_text(entry).rstrip("/").split("/")[-1]
