from __future__ import annotations

import os
import shlex
from pathlib import Path

from .base import BaseInstalledAdapter, ExecCommand


class AiderAdapter(BaseInstalledAdapter):
    name = "aider"

    @property
    def install_template_path(self) -> Path:
        return Path(__file__).parent / "install-aider.sh"

    def create_run_commands(self, instruction: str, *, task, seed: int) -> list[ExecCommand]:
        escaped_instruction = shlex.quote(instruction)
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("model_name must be in format provider/model_name")
        provider, model = self.model_name.split("/", 1)
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        elif provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        if not api_key:
            raise ValueError(f"Missing API key for provider: {provider}")
        env = {"AIDER_API_KEY": f"{provider}={api_key}"}
        return [
            ExecCommand(
                cmd=f"aider --message {escaped_instruction} --yes --model {model} 2>&1 | tee /logs/agent/aider.txt",
                env=env,
            )
        ]
