from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .env import ExecResult, ToolExecutor


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ToolEvent:
    t: str
    tool: str
    args: dict[str, Any]
    result: Any
    error: str | None = None


ReplayMode = Literal["off", "record", "replay"]


class RecordingToolExecutor:
    def __init__(self, inner: ToolExecutor, *, log_path: Path):
        self.inner = inner
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("a", encoding="utf-8")

    def _write(self, ev: ToolEvent) -> None:
        self._fh.write(
            json.dumps(
                {"t": ev.t, "tool": ev.tool, "args": ev.args, "result": ev.result, "error": ev.error},
                default=str,
            )
        )
        self._fh.write("\n")
        self._fh.flush()

    async def exec(self, cmd: str, *, timeout_s: float | None = None) -> ExecResult:
        try:
            res = await self.inner.exec(cmd, timeout_s=timeout_s)
            self._write(
                ToolEvent(
                    t=_utc(),
                    tool="exec",
                    args={"cmd": cmd, "timeout_s": timeout_s},
                    result={
                        "stdout": res.stdout,
                        "stderr": res.stderr,
                        "exit_code": res.exit_code,
                        "duration_ms": res.duration_ms,
                    },
                )
            )
            return res
        except Exception as e:  # pragma: no cover
            self._write(ToolEvent(t=_utc(), tool="exec", args={"cmd": cmd, "timeout_s": timeout_s}, result=None, error=str(e)))
            raise

    async def read_file(self, path: str) -> str:
        try:
            out = await self.inner.read_file(path)
            self._write(ToolEvent(t=_utc(), tool="read_file", args={"path": path}, result=out))
            return out
        except Exception as e:  # pragma: no cover
            self._write(ToolEvent(t=_utc(), tool="read_file", args={"path": path}, result=None, error=str(e)))
            raise

    async def write_file(self, path: str, content: str) -> None:
        try:
            await self.inner.write_file(path, content)
            self._write(ToolEvent(t=_utc(), tool="write_file", args={"path": path}, result={"ok": True}))
        except Exception as e:  # pragma: no cover
            self._write(ToolEvent(t=_utc(), tool="write_file", args={"path": path}, result=None, error=str(e)))
            raise

    async def list_dir(self, path: str) -> list[str]:
        try:
            out = await self.inner.list_dir(path)
            self._write(ToolEvent(t=_utc(), tool="list_dir", args={"path": path}, result=out))
            return out
        except Exception as e:  # pragma: no cover
            self._write(ToolEvent(t=_utc(), tool="list_dir", args={"path": path}, result=None, error=str(e)))
            raise

    async def call(self, tool_name: str, args: dict[str, Any]) -> Any:
        try:
            out = await self.inner.call(tool_name, args)
            self._write(ToolEvent(t=_utc(), tool=f"call:{tool_name}", args=args, result=out))
            return out
        except Exception as e:  # pragma: no cover
            self._write(ToolEvent(t=_utc(), tool=f"call:{tool_name}", args=args, result=None, error=str(e)))
            raise

    async def sync_logs(self) -> None:
        sync_logs = getattr(self.inner, "sync_logs", None)
        if callable(sync_logs):
            await sync_logs()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class ReplayingToolExecutor:
    def __init__(self, *, log_path: Path, strict: bool = True):
        self.log_path = log_path
        self.strict = strict
        self._events = self._load(log_path)
        self._i = 0

    def _load(self, p: Path) -> list[dict[str, Any]]:
        if not p.exists():
            raise FileNotFoundError(p)
        out: list[dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Invalid replay log JSON at line {i}: {e}") from e
        return out

    def _next(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if self._i >= len(self._events):
            raise RuntimeError("Replay log exhausted")
        ev = self._events[self._i]
        self._i += 1
        if self.strict:
            if ev.get("tool") != tool or ev.get("args") != args:
                raise RuntimeError(f"Replay mismatch: expected {tool} {args}, got {ev.get('tool')} {ev.get('args')}")
        return ev

    async def exec(self, cmd: str, *, timeout_s: float | None = None) -> ExecResult:
        ev = self._next("exec", {"cmd": cmd, "timeout_s": timeout_s})
        if ev.get("error"):
            raise RuntimeError(ev["error"])
        r = ev.get("result") or {}
        return ExecResult(
            stdout=str(r.get("stdout", "")),
            stderr=str(r.get("stderr", "")),
            exit_code=int(r.get("exit_code", 0)),
            duration_ms=int(r.get("duration_ms", 0)),
        )

    async def read_file(self, path: str) -> str:
        ev = self._next("read_file", {"path": path})
        if ev.get("error"):
            raise RuntimeError(ev["error"])
        return str(ev.get("result", ""))

    async def write_file(self, path: str, content: str) -> None:
        ev = self._next("write_file", {"path": path})
        if ev.get("error"):
            raise RuntimeError(ev["error"])
        return None

    async def list_dir(self, path: str) -> list[str]:
        ev = self._next("list_dir", {"path": path})
        if ev.get("error"):
            raise RuntimeError(ev["error"])
        return list(ev.get("result") or [])

    async def call(self, tool_name: str, args: dict[str, Any]) -> Any:
        ev = self._next(f"call:{tool_name}", args)
        if ev.get("error"):
            raise RuntimeError(ev["error"])
        return ev.get("result")

    async def sync_logs(self) -> None:
        return None
