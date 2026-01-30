from __future__ import annotations

import os
import shlex
from pathlib import Path

from .base import BaseInstalledAdapter, ExecCommand


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
        if "MSWEA_API_KEY" in os.environ:
            env["MSWEA_API_KEY"] = os.environ["MSWEA_API_KEY"]
        else:
            for key in [
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GOOGLE_API_KEY",
                "GEMINI_API_KEY",
            ]:
                if key in os.environ:
                    env["MSWEA_API_KEY"] = os.environ[key]
                    break
        if "OPENAI_API_BASE" in os.environ:
            env["OPENAI_API_BASE"] = os.environ["OPENAI_API_BASE"]
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
