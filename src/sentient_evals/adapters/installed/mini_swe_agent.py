from __future__ import annotations

import shlex
from pathlib import Path

from .base import BaseInstalledAdapter, ExecCommand
from .parsers.mini_swe_agent import parse_mini_swe_agent_trajectory
from ...models import TranscriptEvent


class MiniSweAgentAdapter(BaseInstalledAdapter):
    name = "mini-swe-agent"

    @property
    def install_template_path(self) -> Path:
        return Path(__file__).parent / "install-mini-swe-agent.sh"

    @property
    def _trajectory_path(self) -> str:
        return "/logs/agent/mini-swe-agent.trajectory.json"

    def create_run_commands(self, instruction: str, *, task, seed: int) -> list[ExecCommand]:
        escaped_instruction = shlex.quote(instruction)
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("model_name must be in format provider/model_name")
        env = {"MSWEA_CONFIGURED": "true"}
        if self._env_has("MSWEA_API_KEY"):
            env["MSWEA_API_KEY"] = self._env_get("MSWEA_API_KEY", "")
        else:
            for key in [
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GOOGLE_API_KEY",
                "GEMINI_API_KEY",
            ]:
                if self._env_has(key):
                    env["MSWEA_API_KEY"] = self._env_get(key, "")
                    break
        if self._env_has("OPENAI_API_BASE"):
            env["OPENAI_API_BASE"] = self._env_get("OPENAI_API_BASE", "")
        return [
            ExecCommand(
                cmd=(
                    f"mini -m {self.model_name} -t {escaped_instruction} -y "
                    f"-o {self._trajectory_path} -l 0 --exit-immediately "
                    "2>&1 </dev/null | tee /logs/agent/mini-swe-agent.txt"
                ),
                env=env,
            )
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
        traj_path = trial_dir / "env_logs" / "agent" / "mini-swe-agent.trajectory.json"
        parsed = parse_mini_swe_agent_trajectory(traj_path, instruction=instruction, model_name=self.model_name)
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
