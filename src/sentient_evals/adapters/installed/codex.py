from __future__ import annotations

import shlex
from pathlib import Path

from .base import BaseInstalledAdapter, ExecCommand
from .parsers.codex import parse_codex_session
from ...models import TranscriptEvent


class CodexAdapter(BaseInstalledAdapter):
    name = "codex"

    def __init__(self, *, reasoning_effort: str | None = "high", **kwargs) -> None:
        super().__init__(**kwargs)
        self._reasoning_effort = reasoning_effort

    @property
    def install_template_path(self) -> Path:
        return Path(__file__).parent / "install-codex.sh"

    def create_run_commands(self, instruction: str, *, task, seed: int) -> list[ExecCommand]:
        escaped_instruction = shlex.quote(instruction)
        if not self.model_name:
            raise ValueError("model_name is required")
        model = self.model_name.split("/")[-1]
        api_key = self._env_get("OPENAI_API_KEY", "")
        if not self._env_has("OPENAI_API_KEY"):
            raise ValueError("codex adapter requires OPENAI_API_KEY environment variable")
        env = {
            "OPENAI_API_KEY": api_key,
            "CODEX_API_KEY": api_key,
            "CODEX_HOME": "/logs/agent",
        }
        reasoning_effort = self._reasoning_effort
        reasoning_flag = f"-c model_reasoning_effort={reasoning_effort} " if reasoning_effort else ""
        return [
            ExecCommand(
                cmd=(
                    "codex exec "
                    "--dangerously-bypass-approvals-and-sandbox "
                    "--skip-git-repo-check "
                    f"--model {model} "
                    "--json "
                    "--enable unified_exec "
                    f"{reasoning_flag}"
                    "-- "
                    f"{escaped_instruction} "
                    "2>&1 </dev/null | tee /logs/agent/codex.txt"
                ),
                env=env,
            ),
        ]

    async def parse_run_artifacts(
        self,
        *,
        task,
        instruction: str,
        results,
        artifacts,
    ) -> list[TranscriptEvent]:
        trial_dir = artifacts.base_dir
        agent_logs = trial_dir / "env_logs" / "agent"
        parsed = parse_codex_session(agent_logs, instruction=instruction)
        if parsed is None:
            return await super().parse_run_artifacts(
                task=task, instruction=instruction, results=results, artifacts=artifacts
            )
        parsed_dir = artifacts.agent().scoped("parsed")
        parsed_dir.write_json("metrics.json", parsed.metrics)
        if parsed.extra:
            parsed_dir.write_json("extra.json", parsed.extra)
        out = list(parsed.events)
        if parsed.metrics:
            out.append(TranscriptEvent(kind="metric", role="system", content="metrics", metrics=parsed.metrics))
        return out
