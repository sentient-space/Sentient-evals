from __future__ import annotations

import asyncio
import base64
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .environments.base import BaseEnvironment


@dataclass(frozen=True)
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int


class ToolExecutor(Protocol):
    async def exec(self, cmd: str, *, timeout_s: float | None = None) -> ExecResult:
        ...

    async def read_file(self, path: str) -> str:
        ...

    async def write_file(self, path: str, content: str) -> None:
        ...

    async def list_dir(self, path: str) -> list[str]:
        ...

    async def call(self, tool_name: str, args: dict[str, Any]) -> Any:
        ...

    async def sync_logs(self) -> None:
        ...


class LocalToolExecutor:
    def __init__(self, *, root: Path):
        self.root = root

    def _abs(self, path: str) -> Path:
        p = (self.root / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError("Path escapes executor root")
        return p

    async def exec(self, cmd: str, *, timeout_s: float | None = None) -> ExecResult:
        loop = asyncio.get_running_loop()

        def _run() -> ExecResult:
            started = loop.time()
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=self.root,
                text=True,
                capture_output=True,
                timeout=timeout_s,
            )
            dur_ms = int((loop.time() - started) * 1000)
            return ExecResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=int(proc.returncode),
                duration_ms=dur_ms,
            )

        return await loop.run_in_executor(None, _run)

    async def read_file(self, path: str) -> str:
        return self._abs(path).read_text(encoding="utf-8")

    async def write_file(self, path: str, content: str) -> None:
        p = self._abs(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    async def list_dir(self, path: str) -> list[str]:
        p = self._abs(path)
        if not p.exists():
            return []
        return sorted([x.name for x in p.iterdir()])

    async def call(self, tool_name: str, args: dict[str, Any]) -> Any:
        if tool_name == "exec":
            res = await self.exec(str(args.get("cmd", "")), timeout_s=args.get("timeout_s"))
            return {
                "stdout": res.stdout,
                "stderr": res.stderr,
                "exit_code": res.exit_code,
                "duration_ms": res.duration_ms,
            }
        if tool_name == "read_file":
            return await self.read_file(str(args.get("path", "")))
        if tool_name == "write_file":
            await self.write_file(str(args.get("path", "")), str(args.get("content", "")))
            return {"ok": True}
        if tool_name == "list_dir":
            return await self.list_dir(str(args.get("path", "")))
        raise ValueError(f"Unknown tool: {tool_name}")

    async def sync_logs(self) -> None:
        return None


class EnvironmentToolExecutor:
    """
    Adapter-facing ToolExecutor backed by a BaseEnvironment.

    This is intentionally minimal and implemented via shell commands executed
    inside the environment. Container images should include a POSIX shell and
    `base64` for write support.
    """

    def __init__(self, env: BaseEnvironment):
        self.env = env

    async def exec(self, cmd: str, *, timeout_s: float | None = None) -> ExecResult:
        return await self.env.exec(cmd, timeout_s=timeout_s)

    async def read_file(self, path: str) -> str:
        q = shlex.quote(path)
        res = await self.env.exec(f"sh -lc 'cat {q}'")
        if res.exit_code != 0:
            raise RuntimeError(res.stderr.strip() or f"read_file failed: {path}")
        return res.stdout

    async def write_file(self, path: str, content: str) -> None:
        q = shlex.quote(path)
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        qb64 = shlex.quote(b64)
        cmd = f"sh -lc 'mkdir -p \"$(dirname -- {q})\" && printf %s {qb64} | base64 -d > {q}'"
        res = await self.env.exec(cmd)
        if res.exit_code != 0:
            raise RuntimeError(res.stderr.strip() or f"write_file failed: {path}")

    async def list_dir(self, path: str) -> list[str]:
        q = shlex.quote(path)
        res = await self.env.exec(f"sh -lc 'ls -1 {q} 2>/dev/null || true'")
        return [line for line in res.stdout.splitlines() if line.strip()]

    async def call(self, tool_name: str, args: dict[str, Any]) -> Any:
        if tool_name == "exec":
            res = await self.exec(str(args.get("cmd", "")), timeout_s=args.get("timeout_s"))
            return {
                "stdout": res.stdout,
                "stderr": res.stderr,
                "exit_code": res.exit_code,
                "duration_ms": res.duration_ms,
            }
        if tool_name == "read_file":
            return await self.read_file(str(args.get("path", "")))
        if tool_name == "write_file":
            await self.write_file(str(args.get("path", "")), str(args.get("content", "")))
            return {"ok": True}
        if tool_name == "list_dir":
            return await self.list_dir(str(args.get("path", "")))
        raise ValueError(f"Unknown tool: {tool_name}")

    async def sync_logs(self) -> None:
        await self.env.sync_logs()
