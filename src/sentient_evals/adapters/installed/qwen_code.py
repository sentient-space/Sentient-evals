from __future__ import annotations

import os
import shlex
from pathlib import Path

from .base import BaseInstalledAdapter, ExecCommand


class QwenCodeAdapter(BaseInstalledAdapter):
    name = "qwen-coder"

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._api_key = api_key
        self._base_url = base_url

    @property
    def install_template_path(self) -> Path:
        return Path(__file__).parent / "install-qwen-code.sh"

    def create_run_commands(self, instruction: str, *, task, seed: int) -> list[ExecCommand]:
        escaped_instruction = shlex.quote(instruction)
        env: dict[str, str] = {}
        if self._api_key:
            env["OPENAI_API_KEY"] = self._api_key
        elif "OPENAI_API_KEY" in os.environ:
            env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
        if self.model_name:
            env["OPENAI_MODEL"] = self.model_name
        elif "OPENAI_MODEL" in os.environ:
            env["OPENAI_MODEL"] = os.environ["OPENAI_MODEL"]
        else:
            env["OPENAI_MODEL"] = "qwen3-coder-plus"
        if self._base_url:
            env["OPENAI_BASE_URL"] = self._base_url
        elif "OPENAI_BASE_URL" in os.environ:
            env["OPENAI_BASE_URL"] = os.environ["OPENAI_BASE_URL"]
        return [
            ExecCommand(
                cmd=f"echo {escaped_instruction} | qwen -y 2>&1 | tee /logs/agent/qwen-code.txt",
                env=env,
            )
        ]
