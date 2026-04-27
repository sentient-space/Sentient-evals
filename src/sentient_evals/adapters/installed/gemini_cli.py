from __future__ import annotations

import shlex
from pathlib import Path

from .base import BaseInstalledAdapter, ExecCommand
from .capabilities import AdapterCapabilities
from .parsers.gemini_cli import parse_gemini_trajectory
from .parsers.structured_streams import parse_gemini_stream
from ...models import TranscriptEvent


class GeminiCliAdapter(BaseInstalledAdapter):
    name = "gemini-cli"
    capabilities = AdapterCapabilities(
        capture_mode="native_stream_json",
        tool_call_support="best_effort",
        metrics_support="best_effort",
        trajectory_confidence="normalized",
        notes="Uses documented Gemini CLI stream-json output, with copied session trajectory as fallback.",
    )

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
            if self._env_has(var):
                env[var] = self._env_get(var, "")
        return [
            ExecCommand(
                cmd=(
                    f"gemini -p {escaped_instruction} -y -m {model} --output-format stream-json "
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
        parsed = parse_gemini_stream(agent_logs, instruction=instruction)
        parser_name = "gemini-stream"
        raw_artifacts = ["agent/gemini-cli.txt"]
        if parsed is None:
            parsed = parse_gemini_trajectory(agent_logs, instruction=instruction)
            parser_name = "gemini-session-trajectory"
            raw_artifacts = ["agent/gemini-cli.trajectory.json"]
        if parsed is None:
            return await super().parse_run_artifacts(
                task=task, instruction=instruction, results=results, artifacts=artifacts
            )
        return self._events_from_parse_result(
            artifacts,
            parsed,
            parser_name=parser_name,
            raw_artifacts=raw_artifacts,
        )
