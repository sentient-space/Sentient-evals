from __future__ import annotations

import os
import shlex
from pathlib import Path

from .base import BaseInstalledAdapter, ExecCommand
from .parsers.claude_code import parse_claude_code_session
from ...models import TranscriptEvent


class ClaudeCodeAdapter(BaseInstalledAdapter):
    name = "claude-code"

    ALLOWED_TOOLS = [
        "Bash",
        "Edit",
        "Write",
        "Read",
        "Glob",
        "Grep",
        "LS",
        "WebFetch",
        "NotebookEdit",
        "NotebookRead",
        "TodoRead",
        "TodoWrite",
        "Agent",
        "Skill",
        "SlashCommand",
        "Task",
        "WebSearch",
    ]

    def __init__(self, *, max_thinking_tokens: int | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_thinking_tokens = max_thinking_tokens

    @property
    def install_template_path(self) -> Path:
        return Path(__file__).parent / "install-claude-code.sh"

    def create_run_commands(self, instruction: str, *, task, seed: int) -> list[ExecCommand]:
        escaped_instruction = shlex.quote(instruction)
        env = {
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
            "ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL", None),
            "CLAUDE_CODE_OAUTH_TOKEN": os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", ""),
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS", None),
            "FORCE_AUTO_BACKGROUND_TASKS": "1",
            "ENABLE_BACKGROUND_TASKS": "1",
        }
        env = {k: v for k, v in env.items() if v}
        if self.model_name:
            if "ANTHROPIC_BASE_URL" in env:
                env["ANTHROPIC_MODEL"] = self.model_name
            else:
                env["ANTHROPIC_MODEL"] = self.model_name.split("/")[-1]
        elif "ANTHROPIC_MODEL" in os.environ:
            env["ANTHROPIC_MODEL"] = os.environ["ANTHROPIC_MODEL"]
        if "ANTHROPIC_BASE_URL" in env and "ANTHROPIC_MODEL" in env:
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = env["ANTHROPIC_MODEL"]
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = env["ANTHROPIC_MODEL"]
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = env["ANTHROPIC_MODEL"]
            env["CLAUDE_CODE_SUBAGENT_MODEL"] = env["ANTHROPIC_MODEL"]
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        max_thinking_tokens = self._max_thinking_tokens
        if max_thinking_tokens is not None:
            env["MAX_THINKING_TOKENS"] = str(max_thinking_tokens)
        elif "MAX_THINKING_TOKENS" in os.environ:
            env["MAX_THINKING_TOKENS"] = os.environ["MAX_THINKING_TOKENS"]
        env["CLAUDE_CONFIG_DIR"] = "/logs/agent/sessions"
        return [
            ExecCommand(
                cmd=(
                    "mkdir -p $CLAUDE_CONFIG_DIR/debug $CLAUDE_CONFIG_DIR/projects/-app "
                    "$CLAUDE_CONFIG_DIR/shell-snapshots $CLAUDE_CONFIG_DIR/statsig "
                    "$CLAUDE_CONFIG_DIR/todos && "
                    "if [ -d ~/.claude/skills ]; then "
                    "cp -r ~/.claude/skills $CLAUDE_CONFIG_DIR/skills 2>/dev/null || true; "
                    "fi"
                ),
                env=env,
            ),
            ExecCommand(
                cmd=(
                    f"claude --verbose --output-format stream-json -p {escaped_instruction} "
                    f"--allowedTools {' '.join(self.ALLOWED_TOOLS)} 2>&1 </dev/null | "
                    "tee /logs/agent/claude-code.txt"
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
        parsed = parse_claude_code_session(agent_logs, instruction=instruction)
        if parsed is None:
            return await super().parse_run_artifacts(
                task=task, instruction=instruction, results=results, artifacts=artifacts
            )
        parsed_dir = artifacts.agent().scoped("parsed")
        parsed_dir.write_json("metrics.json", parsed.metrics)
        if parsed.tool_definitions is not None:
            parsed_dir.write_json("tool_definitions.json", parsed.tool_definitions)
        if parsed.extra:
            parsed_dir.write_json("extra.json", parsed.extra)
        out = list(parsed.events)
        if parsed.metrics:
            out.append(TranscriptEvent(kind="metric", role="system", content="metrics", metrics=parsed.metrics))
        return out
