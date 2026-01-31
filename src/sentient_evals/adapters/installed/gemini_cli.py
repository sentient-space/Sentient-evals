from __future__ import annotations

import os
import shlex
from pathlib import Path

from .base import BaseInstalledAdapter, ExecCommand
from .parsers.gemini_cli import parse_gemini_trajectory
from ...models import TranscriptEvent


class GeminiCliAdapter(BaseInstalledAdapter):
    name = "gemini-cli"

    @property
    def install_template_path(self) -> Path:
        return Path(__file__).parent / "install-gemini-cli.sh"

    def create_run_commands(self, instruction: str, *, task, seed: int) -> list[ExecCommand]:
        escaped_instruction = shlex.quote(instruction)
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("model_name must be in format provider/model_name")
        model = self.model_name.split("/")[-1]
        env: dict[str, str] = {}
        auth_vars = [
            "GEMINI_API_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
            "GOOGLE_GENAI_USE_VERTEXAI",
            "GOOGLE_API_KEY",
        ]
        for var in auth_vars:
            if var in os.environ:
                env[var] = os.environ[var]
        return [
            ExecCommand(
                cmd=(
                    f"gemini -p {escaped_instruction} -y -m {model} "
                    "2>&1 </dev/null | tee /logs/agent/gemini-cli.txt"
                ),
                env=env,
            )
            ,
            ExecCommand(
                cmd=(
                    "find ~/.gemini/tmp -type f -name 'session-*.json' 2>/dev/null | "
                    "head -n 1 | xargs -r -I{} cp {} /logs/agent/gemini-cli.trajectory.json || true"
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
        parsed = parse_gemini_trajectory(agent_logs, instruction=instruction)
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
