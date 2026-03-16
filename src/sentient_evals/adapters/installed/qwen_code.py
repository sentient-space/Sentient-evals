from __future__ import annotations

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
        elif self._env_has("OPENAI_API_KEY"):
            env["OPENAI_API_KEY"] = self._env_get("OPENAI_API_KEY", "")
        if self.model_name:
            env["OPENAI_MODEL"] = self.model_name
        elif self._env_has("OPENAI_MODEL"):
            env["OPENAI_MODEL"] = self._env_get("OPENAI_MODEL", "")
        else:
            env["OPENAI_MODEL"] = "qwen3-coder-plus"
        if self._base_url:
            env["OPENAI_BASE_URL"] = self._base_url
        elif self._env_has("OPENAI_BASE_URL"):
            env["OPENAI_BASE_URL"] = self._env_get("OPENAI_BASE_URL", "")
        return [
            ExecCommand(
                cmd=f"echo {escaped_instruction} | qwen -y 2>&1 | tee /logs/agent/qwen-code.txt",
                env=env,
            )
        ]
