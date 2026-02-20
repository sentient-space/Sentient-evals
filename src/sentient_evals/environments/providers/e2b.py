from __future__ import annotations

import asyncio
import os
import shlex
from pathlib import Path
from typing import Any

from sentient_evals.env import ExecResult

from .base import CloudSandboxProvider, SandboxCapabilities, SandboxCreateParams


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        return "".join(_normalize_text(v) for v in value)
    if isinstance(value, tuple):
        return "".join(_normalize_text(v) for v in value)
    return str(value)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_api_key(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip().strip("'").strip('"')
    if not value:
        return None
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    return value or None


class E2BProvider(CloudSandboxProvider):
    name = "e2b"
    _preferred_user = "root"
    _default_sandbox_timeout_sec = 3_600
    capabilities = SandboxCapabilities(
        supports_network_toggle=True,
        supports_snapshots=False,
        supports_gpus=False,
        supports_attach=False,
        supports_stop=True,
        max_upload_batch=100,
    )

    async def create(self, params: SandboxCreateParams) -> Any:
        Sandbox = self._sandbox_cls()
        template = params.snapshot or params.image
        timeout = self._default_sandbox_timeout_sec
        allow_internet = not bool(params.network_block_all)
        api_key = _normalize_api_key(os.environ.get("E2B_API_KEY"))
        if api_key:
            # Keep env clean for SDK internals that may read directly from os.environ.
            os.environ["E2B_API_KEY"] = api_key

        create_kwargs: dict[str, Any] = {
            "timeout": timeout,
            "allow_internet_access": allow_internet,
            "metadata": {"provider": "sentient-evals"},
        }
        if api_key:
            create_kwargs["api_key"] = api_key
        create_kwargs["template"] = template or "base"

        if params.dockerfile is not None and template is None:
            create_kwargs["metadata"]["dockerfile_ignored"] = str(params.dockerfile)

        if params.resources is not None:
            if params.resources.cpus is not None:
                create_kwargs["metadata"]["cpus"] = str(params.resources.cpus)
            if params.resources.memory_mb is not None:
                create_kwargs["ram_mb"] = int(params.resources.memory_mb)
                create_kwargs["metadata"]["memory_mb"] = str(params.resources.memory_mb)
            if params.resources.storage_mb is not None:
                create_kwargs["metadata"]["storage_mb"] = str(params.resources.storage_mb)
            if params.resources.gpus is not None:
                create_kwargs["metadata"]["gpus"] = str(params.resources.gpus)

        for creator_name in ("beta_create", "create"):
            creator = getattr(Sandbox, creator_name, None)
            if callable(creator):
                try:
                    return await self._invoke(creator, **create_kwargs)
                except TypeError:
                    continue
                except Exception as exc:
                    self._raise_friendly_auth_error(exc)

        try:
            return await self._invoke(Sandbox, **create_kwargs)
        except TypeError:
            sandbox = await self._invoke(Sandbox)
            if timeout is not None:
                await self._set_timeout(sandbox, timeout)
            return sandbox
        except Exception as exc:
            self._raise_friendly_auth_error(exc)

    async def delete(self, sandbox: Any) -> None:
        await self._kill_or_close(sandbox)

    async def stop(self, sandbox: Any) -> None:
        await self._kill_or_close(sandbox)

    async def exec(
        self,
        sandbox: Any,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        loop = asyncio.get_running_loop()
        started = loop.time()
        result = await self._run_command(sandbox, command, cwd=cwd, env=env, timeout_s=timeout_s)
        dur_ms = int((loop.time() - started) * 1000)
        stdout = _normalize_text(getattr(result, "stdout", None))
        stderr = _normalize_text(getattr(result, "stderr", None))
        exit_code = _to_int(
            getattr(result, "exit_code", None)
            if getattr(result, "exit_code", None) is not None
            else getattr(result, "return_code", None),
            default=0,
        )
        return ExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code, duration_ms=dur_ms)

    async def upload_file(self, sandbox: Any, source_path: Path, target_path: str) -> None:
        files = self._files_api(sandbox)
        target_parent = str(Path(target_path).parent)
        await self._run_command(sandbox, f"mkdir -p {shlex.quote(target_parent)}", cwd=None, env=None, timeout_s=30)
        data = source_path.read_bytes()
        write = getattr(files, "write", None)
        if not callable(write):
            raise RuntimeError("E2B filesystem API does not expose write()")
        await self._call_with_optional_user(write, target_path, data=data)

    async def upload_dir(self, sandbox: Any, source_dir: Path, target_dir: str) -> None:
        await self._run_command(sandbox, f"mkdir -p {shlex.quote(target_dir)}", cwd=None, env=None, timeout_s=30)
        for p in sorted(source_dir.rglob("*")):
            if p.is_file() and not p.is_symlink():
                rel = p.relative_to(source_dir).as_posix()
                await self.upload_file(sandbox, p, f"{target_dir.rstrip('/')}/{rel}")

    async def download_file(self, sandbox: Any, source_path: str, target_path: Path) -> None:
        files = self._files_api(sandbox)
        read = getattr(files, "read", None)
        if not callable(read):
            raise RuntimeError("E2B filesystem API does not expose read()")
        data = await self._call_with_optional_user(read, source_path, format="bytes")
        if isinstance(data, dict) and "content" in data:
            data = data["content"]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            target_path.write_text(data, encoding="utf-8")
        elif isinstance(data, bytearray):
            target_path.write_bytes(bytes(data))
        elif isinstance(data, bytes):
            target_path.write_bytes(data)
        else:
            target_path.write_text(_normalize_text(data), encoding="utf-8")

    async def download_dir(self, sandbox: Any, source_dir: str, target_dir: Path) -> None:
        result = await self._run_command(
            sandbox,
            f"cd {shlex.quote(source_dir)} && find . -type f -print",
            cwd=None,
            env=None,
            timeout_s=60,
        )
        stdout = _normalize_text(getattr(result, "stdout", ""))
        target_dir.mkdir(parents=True, exist_ok=True)
        for line in stdout.splitlines():
            rel = line.strip()
            if not rel:
                continue
            rel = rel[2:] if rel.startswith("./") else rel
            if not rel:
                continue
            await self.download_file(sandbox, f"{source_dir.rstrip('/')}/{rel}", target_dir / rel)

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if asyncio.iscoroutine(value):
            return await value
        return value

    @staticmethod
    def _sandbox_cls():
        try:
            from e2b import Sandbox  # type: ignore
        except Exception:
            from e2b_code_interpreter import Sandbox  # type: ignore
        return Sandbox

    @staticmethod
    def _files_api(sandbox: Any) -> Any:
        for attr in ("files", "filesystem", "fs"):
            api = getattr(sandbox, attr, None)
            if api is not None:
                return api
        raise RuntimeError("E2B sandbox does not expose a filesystem API")

    async def _run_command(
        self,
        sandbox: Any,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout_s: float | None,
    ) -> Any:
        request_timeout = float(timeout_s) if timeout_s is not None else None
        command_api = getattr(sandbox, "commands", None)
        if command_api is not None:
            run_cmd = getattr(command_api, "run", None) or getattr(command_api, "run_cmd", None)
            if callable(run_cmd):
                result = await self._call_with_optional_user(
                    run_cmd,
                    command,
                    cwd=cwd,
                    envs=env,
                    timeout=request_timeout,
                    request_timeout=request_timeout,
                    background=False,
                )
                return result

        run = getattr(sandbox, "run", None)
        if callable(run):
            result = await self._call_with_optional_user(
                run,
                command,
                cwd=cwd,
                envs=env,
                timeout=request_timeout,
                request_timeout=request_timeout,
                background=False,
            )
            return result

        process = getattr(sandbox, "process", None)
        if process is not None and callable(getattr(process, "exec", None)):
            result = await self._call_with_optional_user(
                process.exec,
                command=command,
                cwd=cwd,
                env=env,
                timeout=request_timeout,
            )
            return result

        raise RuntimeError("E2B sandbox does not expose a supported command execution API")

    async def _set_timeout(self, sandbox: Any, timeout: int) -> None:
        set_timeout = getattr(sandbox, "set_timeout", None)
        if callable(set_timeout):
            await self._invoke(set_timeout, timeout=timeout)

    async def _kill_or_close(self, sandbox: Any) -> None:
        for name in ("kill", "close", "shutdown", "stop"):
            fn = getattr(sandbox, name, None)
            if callable(fn):
                await self._invoke(fn)
                return

    async def _call_with_optional_user(self, fn, *args, **kwargs):
        kwargs_with_user = dict(kwargs)
        kwargs_with_user["user"] = self._preferred_user
        try:
            return await self._invoke(fn, *args, **kwargs_with_user)
        except TypeError:
            return await self._invoke(fn, *args, **kwargs)

    async def _invoke(self, fn, *args, **kwargs):
        if asyncio.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        result = await asyncio.to_thread(fn, *args, **kwargs)
        return await self._maybe_await(result)

    @staticmethod
    def _raise_friendly_auth_error(exc: Exception) -> None:
        message = str(exc).lower()
        if "authorization header is malformed" in message:
            raise RuntimeError(
                "E2B auth failed: malformed authorization header. "
                "Set E2B_API_KEY to the raw API key value (no 'Bearer ' prefix)."
            ) from exc
        raise exc
