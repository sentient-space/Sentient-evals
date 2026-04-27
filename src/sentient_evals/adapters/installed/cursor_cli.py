from __future__ import annotations

import shlex
from pathlib import Path

from .base import BaseInstalledAdapter, ExecCommand
from .capabilities import AdapterCapabilities
from .parsers.cursor_cli import parse_cursor_cli_stream
from ...models import TranscriptEvent


class CursorCliAdapter(BaseInstalledAdapter):
    name = "cursor-cli"
    capabilities = AdapterCapabilities(
        capture_mode="native_stream_json",
        tool_call_support="guaranteed",
        metrics_support="best_effort",
        trajectory_confidence="normalized",
        notes="Uses cursor-agent --print --output-format stream-json and parses documented NDJSON events.",
    )

    @property
    def install_template_path(self) -> Path:
        return Path(__file__).parent / "install-cursor-cli.sh"

    def create_run_commands(self, instruction: str, *, task, seed: int) -> list[ExecCommand]:
        escaped_instruction = shlex.quote(instruction)
        if not self.model_name:
            raise ValueError("model_name is required")
        model = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
        if not self._env_has("CURSOR_API_KEY"):
            raise ValueError("CURSOR_API_KEY environment variable is required")
        env = {"CURSOR_API_KEY": self._env_get("CURSOR_API_KEY", "")}
        return [
            ExecCommand(
                cmd=(
                    'PATH="$HOME/.local/bin:$PATH"; '
                    'AGENT_BIN="$(command -v agent || command -v cursor-agent || true)"; '
                    '[ -n "$AGENT_BIN" ] || { echo "agent binary not found"; exit 127; }; '
                    'AGENT_HELP="$("$AGENT_BIN" --help 2>/dev/null || true)"; '
                    'EXTRA_FLAGS=""; '
                    'printf "%s" "$AGENT_HELP" | grep -q -- "--force" && EXTRA_FLAGS="$EXTRA_FLAGS --force"; '
                    'printf "%s" "$AGENT_HELP" | grep -q -- "--trust" && EXTRA_FLAGS="$EXTRA_FLAGS --trust"; '
                    'printf "%s" "$AGENT_HELP" | grep -q -- "--yolo" && EXTRA_FLAGS="$EXTRA_FLAGS --yolo"; '
                    '"$AGENT_BIN" --print --output-format stream-json '
                    '${EXTRA_FLAGS} '
                    f"--model {shlex.quote(model)} "
                    '--api-key "$CURSOR_API_KEY" '
                    f"{escaped_instruction} "
                    "2>&1 | tee /logs/agent/cursor-cli.txt"
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
        agent_logs = trial_dir / "env_logs" / "agent"
        parsed = parse_cursor_cli_stream(agent_logs, instruction=instruction)
        if parsed is None:
            return await super().parse_run_artifacts(
                task=task, instruction=instruction, results=results, artifacts=artifacts
            )
        return self._events_from_parse_result(
            artifacts,
            parsed,
            parser_name="cursor-cli-stream",
            raw_artifacts=["agent/cursor-cli.txt"],
        )
