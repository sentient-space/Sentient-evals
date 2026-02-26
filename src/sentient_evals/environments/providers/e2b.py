from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shlex
from pathlib import Path
from typing import Any

from sentient_evals.env import ExecResult

from .base import CloudSandboxProvider, SandboxCapabilities, SandboxCreateParams

try:
    from e2b import AsyncTemplate as _E2BAsyncTemplate
    from e2b import Sandbox as _E2BSandbox
    from e2b import Template as _E2BTemplate
except ImportError:
    _E2BAsyncTemplate = None
    _E2BTemplate = None
    try:
        from e2b_code_interpreter import Sandbox as _E2BSandbox
    except ImportError:
        _E2BSandbox = None


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


def _looks_like_template_id(value: str) -> bool:
    candidate = value.strip()
    if not candidate:
        return False
    if "/" in candidate or ":" in candidate or "@" in candidate:
        return False
    return True


def _hash_path_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        with path.open("rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def _strip_dockerfile_comments(dockerfile: Path) -> str:
    try:
        raw = dockerfile.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = dockerfile.read_text(encoding="utf-8", errors="replace")
    kept_lines = [line for line in raw.splitlines() if not line.lstrip().startswith("#")]
    content = "\n".join(kept_lines).strip()
    return f"{content}\n" if content else ""


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
        template = await self._resolve_template(params)
        timeout = self._default_sandbox_timeout_sec
        allow_internet = not bool(params.network_block_all)
        api_key = _normalize_api_key(os.environ.get("E2B_API_KEY"))
        if api_key:
            os.environ["E2B_API_KEY"] = api_key

        create_kwargs: dict[str, Any] = {
            "timeout": timeout,
            "allow_internet_access": allow_internet,
            "metadata": {"provider": "sentient-evals"},
        }
        if api_key:
            create_kwargs["api_key"] = api_key
        create_kwargs["template"] = template

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
        try:
            result = await self._run_command(sandbox, command, cwd=cwd, env=env, timeout_s=timeout_s)
            stdout = _normalize_text(getattr(result, "stdout", None))
            stderr = _normalize_text(getattr(result, "stderr", None))
            exit_code = _to_int(
                getattr(result, "exit_code", None)
                if getattr(result, "exit_code", None) is not None
                else getattr(result, "return_code", None),
                default=0,
            )
        except Exception as exc:
            coerced = self._coerce_nonzero_exec_exception(exc)
            if coerced is None:
                raise
            stdout, stderr, exit_code = coerced
        dur_ms = int((loop.time() - started) * 1000)
        return ExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code, duration_ms=dur_ms)

    @staticmethod
    def _coerce_nonzero_exec_exception(exc: Exception) -> tuple[str, str, int] | None:
        raw = str(exc)
        message = raw.strip()

        code = _to_int(getattr(exc, "exit_code", None), default=-1)
        stdout = _normalize_text(getattr(exc, "stdout", ""))
        stderr = _normalize_text(getattr(exc, "stderr", ""))
        if code >= 0:
            if not stderr:
                stderr = message
            return stdout, stderr, code

        match = re.search(r"Command exited with code\s+(-?\d+)(?:\s+and error:\n(?P<err>.*))?$", message, re.S)
        if not match:
            return None

        parsed_code = _to_int(match.group(1), default=-1)
        if parsed_code < 0:
            return None

        parsed_err = (match.group("err") or "").strip()
        return "", parsed_err or message, parsed_code

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

    async def _resolve_template(self, params: SandboxCreateParams) -> str:
        if params.snapshot:
            return params.snapshot

        image = (params.image or "").strip() or None
        if image and _looks_like_template_id(image):
            return image

        should_build = bool(image) or params.dockerfile is not None
        if not should_build:
            return "base"

        return await self._build_or_reuse_template(params, source_image=image)

    async def _build_or_reuse_template(
        self,
        params: SandboxCreateParams,
        *,
        source_image: str | None,
    ) -> str:
        template_cls, async_template_cls = self._template_classes()
        if template_cls is None or async_template_cls is None:
            raise RuntimeError(
                "E2B template build requires the e2b SDK Template APIs. "
                "Install/update `e2b` and provide a valid template id in [environment].image as fallback."
            )

        alias = self._template_alias(params, source_image=source_image)
        exists = await self._template_alias_exists(async_template_cls, alias)
        if params.force_build or not exists:
            template = self._create_template_definition(
                template_cls,
                source_image=source_image,
                dockerfile=params.dockerfile,
            )
            build_kwargs: dict[str, Any] = {
                "template": template,
                "alias": alias,
            }
            if params.resources is not None:
                if params.resources.cpus is not None:
                    build_kwargs["cpu_count"] = params.resources.cpus
                if params.resources.memory_mb is not None:
                    build_kwargs["memory_mb"] = int(params.resources.memory_mb)
            await self._invoke(async_template_cls.build, **build_kwargs)
        return alias

    def _template_alias(self, params: SandboxCreateParams, *, source_image: str | None) -> str:
        if source_image:
            digest = hashlib.sha256(f"image:{source_image}".encode("utf-8")).hexdigest()
        elif params.context_dir and params.context_dir.exists():
            digest = _hash_path_tree(params.context_dir)
        elif params.dockerfile and params.dockerfile.exists():
            digest = hashlib.sha256(params.dockerfile.read_bytes()).hexdigest()
        else:
            digest = hashlib.sha256(b"base").hexdigest()
        return f"sentient-e2b-{digest[:16]}"

    def _create_template_definition(
        self,
        template_cls: Any,
        *,
        source_image: str | None,
        dockerfile: Path | None,
    ) -> Any:
        template = template_cls()
        if source_image:
            from_image = getattr(template, "from_image", None)
            if not callable(from_image):
                raise RuntimeError("E2B Template API does not expose from_image()")
            return from_image(image=source_image)
        if dockerfile is None:
            raise RuntimeError("E2B template build requires source image or dockerfile")
        from_dockerfile = getattr(template, "from_dockerfile", None)
        if not callable(from_dockerfile):
            raise RuntimeError("E2B Template API does not expose from_dockerfile()")
        sanitized_content = _strip_dockerfile_comments(dockerfile)
        if sanitized_content:
            try:
                return from_dockerfile(dockerfile_content_or_path=sanitized_content)
            except TypeError:
                pass
            except (FileNotFoundError, OSError):
                pass
        try:
            return from_dockerfile(dockerfile_content_or_path=str(dockerfile))
        except TypeError:
            return from_dockerfile(str(dockerfile))

    async def _template_alias_exists(self, async_template_cls: Any, alias: str) -> bool:
        alias_exists = getattr(async_template_cls, "alias_exists", None)
        if not callable(alias_exists):
            return False
        try:
            return bool(await self._invoke(alias_exists, alias))
        except Exception:
            return False

    @staticmethod
    def _sandbox_cls():
        if _E2BSandbox is None:
            raise RuntimeError(
                "E2B SDK is not installed. Install `e2b` (preferred) or `e2b-code-interpreter`."
            )
        return _E2BSandbox

    @staticmethod
    def _template_classes() -> tuple[Any | None, Any | None]:
        return _E2BTemplate, _E2BAsyncTemplate

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
