from __future__ import annotations

import shlex
from pathlib import Path

from .base import BaseInstalledAdapter, ExecCommand


class ClineCliAdapter(BaseInstalledAdapter):
    name = "cline-cli"

    @property
    def install_template_path(self) -> Path:
        return Path(__file__).parent / "install-cline.sh"

    def create_run_commands(self, instruction: str, *, task, seed: int) -> list[ExecCommand]:
        escaped_instruction = shlex.quote(instruction)
        if not self.model_name or ":" not in self.model_name:
            raise ValueError("model_name must be in format provider:model-id")
        provider, model = self.model_name.split(":", 1)
        if not self._env_has("API_KEY"):
            raise ValueError("API_KEY environment variable is required")
        env = {
            "PROVIDER": provider,
            "API_KEY": self._env_get("API_KEY", ""),
            "MODELID": model,
        }
        valid_providers = [
            "anthropic",
            "openai",
            "openai-compatible",
            "openai-native",
            "openrouter",
            "xai",
            "bedrock",
            "gemini",
            "ollama",
            "cerebras",
            "cline",
            "oca",
            "hicap",
            "nousresearch",
        ]
        if provider not in valid_providers:
            raise ValueError(f"Invalid provider: {provider}")
        base_url = ""
        if provider == "openai":
            if not self._env_has("BASE_URL"):
                raise ValueError("BASE_URL is required for openai provider")
            env["BASE_URL"] = self._env_get("BASE_URL", "")
            base_url = env["BASE_URL"]
        setup_config_cmd = ExecCommand(
            cmd=(
                "mkdir -p ~/.cline/data && "
                "cat > ~/.cline/data/globalState.json <<EOF\n"
                '{"welcomeViewCompleted": true, "isNewUser": false}\n'
                "EOF"
            ),
            env=env,
        )
        if provider == "openai" and base_url:
            auth_cmd = 'cline auth -p openai -k "$API_KEY" -m "$MODELID" -b "$BASE_URL"'
        else:
            auth_cmd = f'cline auth -p {provider} -k "$API_KEY" -m "$MODELID"'
        run_cmd = ExecCommand(
            cmd=(
                'export NVM_DIR="$HOME/.nvm" && '
                '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" && '
                "nvm use 22 && "
                f"{auth_cmd} && "
                f"cline -F plain -o {escaped_instruction} 2>&1 | tee /logs/agent/cline.txt; "
                "EXIT_CODE=$?; "
                "cline instance kill -a || true; "
                "exit $EXIT_CODE"
            ),
            env=env,
        )
        return [setup_config_cmd, run_cmd]
