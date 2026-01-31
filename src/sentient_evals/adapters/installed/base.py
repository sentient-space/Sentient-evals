from __future__ import annotations

import json
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from jinja2 import Environment, StrictUndefined

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
   
    env = Environment(undefined=StrictUndefined)
    return env.from_string(template).render(**variables)


class BaseInstalledAdapter:
    name = "installed_base"
    version: str | None = None
    model_name: str | None = None
    install_timeout_s: float | None = None
    run_timeout_s: float | None = None

    _logger = logging.getLogger(__name__)

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
        return transcript

    def build_outcome(self, transcript: Sequence[TranscriptEvent]) -> Outcome:
        answer = ""
        for ev in reversed(transcript):
            if ev.kind == "message" and ev.role == "assistant":
                answer = ev.content or ""
                break
        
        agg: dict[str, float] = {}
        for ev in transcript:
            if not ev.metrics:
                continue
            for k, v in ev.metrics.items():
                try:
                    fv = float(v)
                except Exception:
                    continue
                # Prefer max to avoid double-counting if logs include totals repeatedly.
                agg[k] = max(agg.get(k, 0.0), fv)
        if "prompt_tokens" in agg or "completion_tokens" in agg:
            agg["total_tokens"] = float(agg.get("prompt_tokens", 0.0) + agg.get("completion_tokens", 0.0))
        if "cost_usd" in agg:
            agg["total_cost_usd"] = float(agg["cost_usd"])
        data: dict[str, Any] = {"answer": answer}
        if agg:
            data["metrics"] = agg
        return Outcome(summary="ok", data=data)

    def _render_install_script(self) -> str:
        template = self.install_template_path.read_text(encoding="utf-8")
        try:
            return _render_template(template, self.template_vars())
        except Exception as exc:
            raise RuntimeError(
                f"Failed to render install script template for adapter={self.name} "
                f"template={self.install_template_path}"
            ) from exc

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
        if res.exit_code != 0:
            raise RuntimeError(
                f"Install failed for adapter={self.name} (exit_code={res.exit_code}). "
                f"See trial artifacts under: {install_dir.base_dir}"
            )

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
            if res.exit_code != 0:
                raise RuntimeError(
                    f"Agent command failed for adapter={self.name} cmd_index={idx} "
                    f"(exit_code={res.exit_code}). See: {cmd_artifacts.base_dir}"
                )
        transcript = await self.parse_run_artifacts(
            task=task, instruction=normalized_instruction, results=results, artifacts=artifacts
        )
        return transcript, self.build_outcome(transcript)
