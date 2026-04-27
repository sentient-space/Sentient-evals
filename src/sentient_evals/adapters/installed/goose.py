from __future__ import annotations

from pathlib import Path

from .base import BaseInstalledAdapter, ExecCommand
from .capabilities import AdapterCapabilities
from .parsers.structured_streams import parse_goose_stream


class GooseAdapter(BaseInstalledAdapter):
    name = "goose"
    capabilities = AdapterCapabilities(
        capture_mode="native_stream_json",
        tool_call_support="best_effort",
        metrics_support="best_effort",
        trajectory_confidence="normalized",
        notes="Runs goose with stream-json output and parses message/toolRequest/toolResponse events.",
    )

    @property
    def install_template_path(self) -> Path:
        return Path(__file__).parent / "install-goose.sh"

    def _recipe_yaml(self, instruction: str) -> str:
        return (
            "version: \"1.0.0\"\n"
            "title: \"sentient-evals-task\"\n"
            "description: \"sentient eval recipe\"\n"
            "instructions: |\n"
            "  You are given a task and you need to complete it.\n"
            "  You are currently executing in a docker container where you are\n"
            "  being evaluated on a benchmark for LLM agents. Act autonomously.\n"
            "  You will not receive any feedback on your progress, so you must\n"
            "  use your own tools to complete the task without any intervention.\n"
            "prompt: |\n"
            f"  {instruction.replace(chr(10), chr(10) + '  ')}\n"
            "extensions:\n"
            "  - type: builtin\n"
            "    name: developer\n"
            "  - type: platform\n"
            "    name: todo\n"
        )

    def create_run_commands(self, instruction: str, *, task, seed: int) -> list[ExecCommand]:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("model_name must be in format provider/model_name")
        provider, model = self.model_name.split("/", 1)
        env = {"GOOSE_MODEL": model, "GOOSE_PROVIDER": provider}
        if provider == "openai":
            api_key = self._env_get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            env["OPENAI_API_KEY"] = api_key
        elif provider == "anthropic":
            api_key = self._env_get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            env["ANTHROPIC_API_KEY"] = api_key
        elif provider == "databricks":
            host = self._env_get("DATABRICKS_HOST")
            token = self._env_get("DATABRICKS_TOKEN")
            if not host or not token:
                raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN are required")
            env["DATABRICKS_HOST"] = host
            env["DATABRICKS_TOKEN"] = token
        elif provider == "tetrate":
            api_key = self._env_get("TETRATE_API_KEY")
            if not api_key:
                raise ValueError("TETRATE_API_KEY environment variable not set")
            env["TETRATE_API_KEY"] = api_key
            host = self._env_get("TETRATE_HOST")
            if host:
                env["TETRATE_HOST"] = host
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        recipe_yaml = self._recipe_yaml(instruction)
        return [
            ExecCommand(
                cmd=f"cat > ~/sentient-evals-recipe.yaml << 'EOF'\n{recipe_yaml}EOF",
                env=env,
                timeout_s=10,
            ),
            ExecCommand(
                cmd=(
                    'export PATH="/root/.local/bin:$PATH" && '
                    "goose run --recipe ~/sentient-evals-recipe.yaml "
                    "--output-format stream-json "
                    "2>&1 | tee /logs/agent/goose.txt"
                ),
                env=env,
            ),
        ]

    async def parse_run_artifacts(self, *, task, instruction: str, results, artifacts):
        trial_dir = artifacts.base_dir
        agent_logs = trial_dir / "env_logs" / "agent"
        parsed = parse_goose_stream(agent_logs, instruction=instruction, model_name=self.model_name)
        if parsed is None:
            return await super().parse_run_artifacts(
                task=task, instruction=instruction, results=results, artifacts=artifacts
            )
        return self._events_from_parse_result(
            artifacts,
            parsed,
            parser_name="goose-stream",
            raw_artifacts=["agent/goose.txt"],
        )
