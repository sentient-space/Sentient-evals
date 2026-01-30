from __future__ import annotations

import os
import shlex
from pathlib import Path

from .base import BaseInstalledAdapter, ExecCommand


class CursorCliAdapter(BaseInstalledAdapter):
    name = "cursor-cli"

    @property
    def install_template_path(self) -> Path:
        return Path(__file__).parent / "install-cursor-cli.sh"

    def create_run_commands(self, instruction: str, *, task, seed: int) -> list[ExecCommand]:
        escaped_instruction = shlex.quote(instruction)
        if not self.model_name:
            raise ValueError("model_name is required")
        model = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
        if "CURSOR_API_KEY" not in os.environ:
            raise ValueError("CURSOR_API_KEY environment variable is required")
        env = {"CURSOR_API_KEY": os.environ["CURSOR_API_KEY"]}
        return [
            ExecCommand(
                cmd=(
                    'PATH="$HOME/.local/bin:$PATH"; '
                    'AGENT_BIN="$(command -v agent || command -v cursor-agent || true)"; '
                    '[ -n "$AGENT_BIN" ] || { echo "agent binary not found"; exit 127; }; '
                    '"$AGENT_BIN" --print --output-format text '
                    f"--model {shlex.quote(model)} "
                    '--api-key "$CURSOR_API_KEY" '
                    f"{escaped_instruction} "
                    "2>&1 | tee /logs/agent/cursor-cli.txt"
                ),
                env=env,
            )
        ]
