from __future__ import annotations

import os
import shlex
from pathlib import Path

from .base import BaseInstalledAdapter, ExecCommand


class OpenCodeAdapter(BaseInstalledAdapter):
    name = "opencode"

    @property
    def install_template_path(self) -> Path:
        return Path(__file__).parent / "install-opencode.sh"

    def create_run_commands(self, instruction: str, *, task, seed: int) -> list[ExecCommand]:
        escaped_instruction = shlex.quote(instruction)
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("model_name must be in format provider/model_name")
        provider, _ = self.model_name.split("/", 1)
        keys: list[str] = []
        if provider == "amazon-bedrock":
            keys.extend(["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"])
        elif provider == "anthropic":
            keys.append("ANTHROPIC_API_KEY")
        elif provider == "azure":
            keys.extend(["AZURE_RESOURCE_NAME", "AZURE_API_KEY"])
        elif provider == "deepseek":
            keys.append("DEEPSEEK_API_KEY")
        elif provider == "github-copilot":
            keys.append("GITHUB_TOKEN")
        elif provider == "google":
            keys.extend(
                [
                    "GEMINI_API_KEY",
                    "GOOGLE_GENERATIVE_AI_API_KEY",
                    "GOOGLE_APPLICATION_CREDENTIALS",
                    "GOOGLE_CLOUD_PROJECT",
                    "GOOGLE_CLOUD_LOCATION",
                    "GOOGLE_GENAI_USE_VERTEXAI",
                    "GOOGLE_API_KEY",
                ]
            )
        elif provider == "groq":
            keys.append("GROQ_API_KEY")
        elif provider == "huggingface":
            keys.append("HF_TOKEN")
        elif provider == "llama":
            keys.append("LLAMA_API_KEY")
        elif provider == "mistral":
            keys.append("MISTRAL_API_KEY")
        elif provider == "openai":
            keys.append("OPENAI_API_KEY")
        elif provider == "xai":
            keys.append("XAI_API_KEY")
        else:
            raise ValueError(f"Unknown provider {provider}")
        env: dict[str, str] = {}
        for key in keys:
            if key in os.environ:
                env[key] = os.environ[key]
        env["OPENCODE_FAKE_VCS"] = "git"
        return [
            ExecCommand(
                cmd=(
                    f"opencode --model {self.model_name} run --format=json {escaped_instruction} "
                    "2>&1 | tee /logs/agent/opencode.txt"
                ),
                env=env,
            )
        ]
