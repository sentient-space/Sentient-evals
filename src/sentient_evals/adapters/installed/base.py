from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ...artifacts import TrialArtifacts
from ...env import ExecResult, ToolExecutor
from ...models import Outcome, Task, TranscriptEvent


@dataclass(frozen=True)
class ExecCommand:
    cmd: str
    timeout_s: float | None = None
    cwd: str | None = None
    env: dict[str, str] | None = None


def _render_template(template: str, variables: dict[str, str | None]) -> str:
    def _replace_if_else(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        truthy = bool(variables.get(key))
        return match.group(2) if truthy else match.group(3)

    def _replace_if(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        truthy = bool(variables.get(key))
        return match.group(2) if truthy else ""

    template = re.sub(
        r"\{\%\s*if\s+([a-zA-Z0-9_]+)\s*\%\}(.*?)\{\%\s*else\s*\%\}(.*?)\{\%\s*endif\s*\%\}",
        _replace_if_else,
        template,
        flags=re.DOTALL,
    )
    template = re.sub(
        r"\{\%\s*if\s+([a-zA-Z0-9_]+)\s*\%\}(.*?)\{\%\s*endif\s*\%\}",
        _replace_if,
        template,
        flags=re.DOTALL,
    )
    for key, value in variables.items():
        template = template.replace(f"{{{{ {key} }}}}", "" if value is None else str(value))
    return template


class BaseInstalledAdapter:
    name = "installed_base"
    version: str | None = None
    model_name: str | None = None
    install_timeout_s: float | None = None
    run_timeout_s: float | None = None

    def __init__(
        self,
        *,
        version: str | None = None,
        model_name: str | None = None,
        install_timeout_s: float | None = None,
        run_timeout_s: float | None = None,
    ) -> None:
        self.version = version
        self.model_name = model_name
        self.install_timeout_s = install_timeout_s
        self.run_timeout_s = run_timeout_s

    @property
    def install_template_path(self) -> Path:
        raise NotImplementedError

    def template_vars(self) -> dict[str, str | None]:
        return {"version": self.version}

    def build_instruction(self, task: Task, instruction: str | None) -> str:
        if instruction and instruction.strip():
            return instruction
        if task.input:
            return json.dumps(task.input, ensure_ascii=False)
        return task.id

    def create_run_commands(self, instruction: str, *, task: Task, seed: int) -> Sequence[ExecCommand]:
        raise NotImplementedError

    async def parse_run_artifacts(
        self,
        *,
        task: Task,
        instruction: str,
        results: Sequence[ExecResult],
        artifacts: TrialArtifacts,
    ) -> list[TranscriptEvent]:
        output = ""
        if results:
            last = results[-1]
            output = "\n".join([part for part in [last.stdout, last.stderr] if part])
        transcript = [
            TranscriptEvent(kind="message", role="user", content=instruction),
            TranscriptEvent(kind="message", role="assistant", content=output.strip()),
        ]
        artifacts.agent().write_text("transcript_preview.txt", output.strip())
        return transcript

    def build_outcome(self, transcript: Sequence[TranscriptEvent]) -> Outcome:
        answer = ""
        for ev in reversed(transcript):
            if ev.kind == "message" and ev.role == "assistant":
                answer = ev.content or ""
                break
        return Outcome(summary="ok", data={"answer": answer})

    def _render_install_script(self) -> str:
        template = self.install_template_path.read_text(encoding="utf-8")
        return _render_template(template, self.template_vars())

    def _format_command(self, cmd: ExecCommand) -> tuple[str, str]:
        parts = []
        safe_parts = []
        if cmd.env:
            for key, value in cmd.env.items():
                if value is None:
                    continue
                quoted = shlex.quote(value)
                parts.append(f"{key}={quoted}")
                safe_parts.append(f"{key}=***")
        if cmd.cwd:
            qcwd = shlex.quote(cmd.cwd)
            parts.append(f"cd {qcwd} &&")
            safe_parts.append(f"cd {qcwd} &&")
        parts.append(cmd.cmd)
        safe_parts.append(cmd.cmd)
        return " ".join(parts), " ".join(safe_parts)

    async def _write_exec_result(self, base: TrialArtifacts, result: ExecResult) -> None:
        base.write_text("stdout.txt", result.stdout)
        base.write_text("stderr.txt", result.stderr)
        base.write_text("exit_code.txt", str(result.exit_code))
        base.write_text("duration_ms.txt", str(result.duration_ms))

    async def install(self, env: ToolExecutor, artifacts: TrialArtifacts) -> None:
        script = self._render_install_script()
        agent_dir = artifacts.agent()
        agent_dir.write_text("install.sh", script)
        # Many installed adapters stream logs to /logs/agent/* and verifiers to /logs/verifier/*.
        # Create them up-front so `tee /logs/agent/foo.txt` doesn't fail.
        await env.exec("sh -lc 'mkdir -p /installed-agent /logs/agent /logs/verifier'")
        await env.write_file("/installed-agent/install.sh", script)
        await env.exec("sh -lc 'chmod +x /installed-agent/install.sh'")
        res = await env.exec("sh -lc /installed-agent/install.sh", timeout_s=self.install_timeout_s)
        install_dir = agent_dir.scoped("install")
        await self._write_exec_result(install_dir, res)

    async def run(
        self,
        task: Task,
        *,
        instruction: str | None,
        seed: int,
        env: ToolExecutor,
        artifacts: TrialArtifacts,
    ) -> tuple[list[TranscriptEvent], Outcome]:
        normalized_instruction = self.build_instruction(task, instruction)
        await self.install(env, artifacts)
        results: list[ExecResult] = []
        for idx, cmd in enumerate(self.create_run_commands(normalized_instruction, task=task, seed=seed)):
            actual_cmd, safe_cmd = self._format_command(cmd)
            cmd_artifacts = artifacts.agent().scoped(f"commands/{idx}")
            cmd_artifacts.write_text("command.txt", safe_cmd)
            res = await env.exec(actual_cmd, timeout_s=cmd.timeout_s or self.run_timeout_s)
            await self._write_exec_result(cmd_artifacts, res)
            results.append(res)
        transcript = await self.parse_run_artifacts(
            task=task, instruction=normalized_instruction, results=results, artifacts=artifacts
        )
        return transcript, self.build_outcome(transcript)
