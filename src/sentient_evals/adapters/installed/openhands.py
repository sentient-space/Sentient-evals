from __future__ import annotations

import shlex
from pathlib import Path

from .base import BaseInstalledAdapter, ExecCommand
from .capabilities import AdapterCapabilities
from .parsers.openhands import parse_openhands_session
from ...models import TranscriptEvent


class OpenHandsAdapter(BaseInstalledAdapter):
    name = "openhands"
    capabilities = AdapterCapabilities(
        capture_mode="native_trajectory",
        tool_call_support="best_effort",
        metrics_support="best_effort",
        trajectory_confidence="normalized",
        notes="Reads OpenHands session events, saved trajectory, and completion logs.",
    )

    def __init__(
        self,
        *,
        disable_tool_calls: bool = False,
        reasoning_effort: str | None = "medium",
        git_version: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._disable_tool_calls = disable_tool_calls
        self._reasoning_effort = reasoning_effort
        self._git_version = git_version

    @property
    def install_template_path(self) -> Path:
        return Path(__file__).parent / "install-openhands.sh"

    def template_vars(self) -> dict[str, str | None]:
        if self._git_version:
            install_cmd = (
                f"uv pip install git+https://github.com/All-Hands-AI/OpenHands.git@{self._git_version}"
            )
        elif self.version:
            install_cmd = f"uv pip install openhands-ai=={self.version}"
        else:
            install_cmd = "uv pip install openhands-ai"
        return {"install_cmd": install_cmd}

    def create_run_commands(self, instruction: str, *, task, seed: int) -> list[ExecCommand]:
        escaped_instruction = shlex.quote(instruction)
        env: dict[str, str] = {}
        if self._env_has("LLM_API_KEY"):
            env["LLM_API_KEY"] = self._env_get("LLM_API_KEY", "")
        else:
            model_name = self.model_name or self._env_get("LLM_MODEL") or self._env_get("ANTHROPIC_MODEL")
            if model_name:
                for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]:
                    if self._env_has(key):
                        env["LLM_API_KEY"] = self._env_get(key, "")
                        break
            if "LLM_API_KEY" not in env:
                raise ValueError("LLM_API_KEY not set and could not infer provider key")
        if self.model_name:
            env["LLM_MODEL"] = self.model_name
        elif self._env_has("LLM_MODEL"):
            env["LLM_MODEL"] = self._env_get("LLM_MODEL", "")
        elif self._env_has("ANTHROPIC_MODEL"):
            env["LLM_MODEL"] = self._env_get("ANTHROPIC_MODEL", "")
        else:
            raise ValueError("LLM_MODEL is required for OpenHands")
        if self._env_has("LLM_BASE_URL"):
            env["LLM_BASE_URL"] = self._env_get("LLM_BASE_URL", "")
        if self._env_has("LLM_API_VERSION"):
            env["LLM_API_VERSION"] = self._env_get("LLM_API_VERSION", "")
        if self._reasoning_effort is not None:
            env["LLM_REASONING_EFFORT"] = str(self._reasoning_effort)
        env["AGENT_ENABLE_PROMPT_EXTENSIONS"] = "false"
        env["AGENT_ENABLE_BROWSING"] = "false"
        env["ENABLE_BROWSER"] = "false"
        env["SANDBOX_ENABLE_AUTO_LINT"] = "true"
        env["SKIP_DEPENDENCY_CHECK"] = "1"
        env["RUN_AS_OPENHANDS"] = "false"
        env["RUNTIME"] = "local"
        env["SAVE_TRAJECTORY_PATH"] = "/logs/agent/openhands.trajectory.json"
        env["FILE_STORE"] = "local"
        env["FILE_STORE_PATH"] = "/logs/agent/"
        env["LLM_LOG_COMPLETIONS"] = "true"
        env["LLM_LOG_COMPLETIONS_FOLDER"] = "/logs/agent/completions/"
        if self._disable_tool_calls:
            env["LLM_NATIVE_TOOL_CALLING"] = "false"
        for key, value in self._env_items():
            if key.startswith("OPENHANDS_"):
                env[key.replace("OPENHANDS_", "")] = value
        cmd = (
            "SANDBOX_VOLUMES=${PWD}:/workspace:rw "
            "/opt/openhands-venv/bin/python -m openhands.core.main "
            f"--task {escaped_instruction} "
            "2>&1 </dev/null | tee /logs/agent/openhands.txt"
        )
        return [ExecCommand(cmd=cmd, env=env)]

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
        parsed = parse_openhands_session(agent_logs, instruction=instruction)
        if parsed is None:
            return await super().parse_run_artifacts(
                task=task, instruction=instruction, results=results, artifacts=artifacts
            )
        return self._events_from_parse_result(
            artifacts,
            parsed,
            parser_name="openhands-session",
            raw_artifacts=["agent/openhands.txt", "agent/openhands.trajectory.json", "agent/sessions/**"],
        )
