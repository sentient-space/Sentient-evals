from __future__ import annotations

import uuid
from pathlib import Path

from .base import BaseInstalledAdapter, ExecCommand
from .parsers.swe_agent import parse_swe_agent_traj
from ...models import TranscriptEvent


class SweAgentAdapter(BaseInstalledAdapter):
    name = "swe-agent"

    @property
    def install_template_path(self) -> Path:
        return Path(__file__).parent / "install-swe-agent.sh"

    def create_run_commands(self, instruction: str, *, task, seed: int) -> list[ExecCommand]:
        if not self.model_name:
            raise ValueError("model_name must be specified for swe-agent")
        env: dict[str, str] = {}
        for key in [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "TOGETHER_API_KEY",
            "SWEAGENT_CONFIG",
        ]:
            if self._env_has(key):
                env[key] = self._env_get(key, "")
        instruction_path = "/logs/agent/problem_statement.md"
        heredoc = f"SENTIENT_INSTRUCTION_{uuid.uuid4().hex}"
        write_instruction_cmd = (
            "mkdir -p /logs/agent\n"
            f"cat > '{instruction_path}' << '{heredoc}'\n"
            f"{instruction}\n"
            f"{heredoc}\n"
        )
        cmd_parts = [
            "sweagent run",
            f"--agent.model.name={self.model_name}",
            f"--problem_statement.path={instruction_path}",
            "--env.deployment.type=local",
            "--output_dir=/logs/agent/swe-agent-output",
            "$(if [ -d /testbed ]; then echo '--env.repo.type=preexisting --env.repo.repo_name=/testbed'; "
            'else echo "--env.repo.path=$(pwd)"; fi)',
        ]
        download_config_cmd = ""
        config_path = "/opt/sweagent-configs/default.yaml"
        if "SWEAGENT_CONFIG" in env:
            config_source = env["SWEAGENT_CONFIG"]
            if config_source.startswith("http://") or config_source.startswith("https://"):
                config_path = "/opt/sweagent-configs/swesmith_infer.yaml"
                download_config_cmd = f"curl -sSL '{config_source}' -o '{config_path}'\n"
            else:
                config_path = env["SWEAGENT_CONFIG"]
        cmd_parts.append(f'--config="{config_path}"')
        if self.model_name.startswith("hosted_vllm/"):
            cmd_parts.extend(
                [
                    "--agent.model.per_instance_cost_limit=0",
                    "--agent.model.total_cost_limit=0",
                    "--agent.model.max_input_tokens=0",
                ]
            )
        if "OPENAI_BASE_URL" in env:
            cmd_parts.append(f"--agent.model.api_base={env['OPENAI_BASE_URL']}")
        command = " ".join(cmd_parts)
        copy_traj_cmd = (
            "TRAJ_FILE=$(find /logs/agent/swe-agent-output -name '*.traj' -print -quit); "
            'if [ -n "$TRAJ_FILE" ]; then '
            'cp "$TRAJ_FILE" /logs/agent/swe-agent.trajectory.json; '
            "fi"
        )
        full_cmd = (
            "set -euo pipefail\n"
            f"{download_config_cmd}"
            f"{write_instruction_cmd}"
            f"{command} 2>&1 | tee /logs/agent/swe-agent.txt\n"
            f"{copy_traj_cmd}\n"
        )
        return [ExecCommand(cmd=full_cmd, env=env)]

    async def parse_run_artifacts(
        self,
        *,
        task,
        instruction: str,
        results,
        artifacts,
    ) -> list[TranscriptEvent]:
        trial_dir = artifacts.base_dir
        traj_path = trial_dir / "env_logs" / "agent" / "swe-agent.trajectory.json"
        parsed = parse_swe_agent_traj(traj_path, instruction=instruction, model_name=self.model_name)
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
