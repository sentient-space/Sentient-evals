from __future__ import annotations

import json
from typing import Any

from sentient_evals.env import ExecResult, ToolExecutor
from sentient_evals.models import TranscriptEvent, ToolCall


def build_instruction_payload(task_input: dict[str, Any]) -> str:
    if task_input:
        return json.dumps(task_input)
    return ""


class ToolExecutorRecorder:
    def __init__(self, env: ToolExecutor, transcript: list[TranscriptEvent]):
        self.env = env
        self.transcript = transcript
        self._call_id = 0

    def _next_call_id(self) -> str:
        self._call_id += 1
        return f"call-{self._call_id}"

    def _record(self, name: str, args: dict[str, Any], observation: Any) -> None:
        call_id = self._next_call_id()
        self.transcript.append(
            TranscriptEvent(
                kind="tool_call",
                tool_call=ToolCall(name=name, args=args),
                observation=observation,
            )
        )

    async def exec(self, cmd: str, *, timeout_s: float | None = None) -> ExecResult:
        result = await self.env.exec(cmd, timeout_s=timeout_s)
        self._record(
            "exec",
            {"cmd": cmd, "timeout_s": timeout_s},
            {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "duration_ms": result.duration_ms,
            },
        )
        return result

    async def read_file(self, path: str) -> str:
        result = await self.env.read_file(path)
        self._record("read_file", {"path": path}, result)
        return result

    async def write_file(self, path: str, content: str) -> None:
        await self.env.write_file(path, content)
        self._record("write_file", {"path": path}, {"ok": True})

    async def list_dir(self, path: str) -> list[str]:
        result = await self.env.list_dir(path)
        self._record("list_dir", {"path": path}, result)
        return result

    async def call(self, tool_name: str, args: dict[str, Any]) -> Any:
        result = await self.env.call(tool_name, args)
        self._record(f"call:{tool_name}", args, result)
        return result
