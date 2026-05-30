from __future__ import annotations

import shlex
from pathlib import Path

from .base import BaseInstalledAdapter, ExecCommand
from .capabilities import AdapterCapabilities
from .parsers.structured_streams import parse_qwen_code_stream


class QwenCodeAdapter(BaseInstalledAdapter):
    name = "qwen-coder"
    capabilities = AdapterCapabilities(
        capture_mode="native_stream_json",
        tool_call_support="best_effort",
        metrics_support="best_effort",
        trajectory_confidence="normalized",
        notes="Uses documented Qwen Code stream-json headless output.",
    )

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
                cmd=f"qwen --yolo --prompt={escaped_instruction} --output-format stream-json 2>&1 | tee /logs/agent/qwen-code.txt",
                env=env,
            )
        ]

    async def parse_run_artifacts(self, *, task, instruction: str, results, artifacts):
        trial_dir = artifacts.base_dir
        agent_logs = trial_dir / "env_logs" / "agent"
        parsed = parse_qwen_code_stream(agent_logs, instruction=instruction)
        if parsed is None:
            return await super().parse_run_artifacts(
                task=task, instruction=instruction, results=results, artifacts=artifacts
            )
        return self._events_from_parse_result(
            artifacts,
            parsed,
            parser_name="qwen-code-stream",
            raw_artifacts=["agent/qwen-code.txt"],
        )
