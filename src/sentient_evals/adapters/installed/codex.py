from __future__ import annotations

import json
import shlex
from pathlib import Path

from .base import BaseInstalledAdapter, ExecCommand
from .capabilities import AdapterCapabilities
from .parsers.codex import parse_codex_exec_output, parse_codex_session
from ...models import TranscriptEvent


class CodexAdapter(BaseInstalledAdapter):
    name = "codex"
    capabilities = AdapterCapabilities(
        capture_mode="session_jsonl",
        tool_call_support="best_effort",
        metrics_support="best_effort",
        trajectory_confidence="normalized",
        notes="Runs codex exec --json and prefers copied Codex session JSONL, with stdout JSON fallback.",
    )

    def __init__(
        self,
        *,
        reasoning_effort: str | None = "high",
        auth_json: str | None = None,
        auth_json_path: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._reasoning_effort = reasoning_effort
        self._auth_json = auth_json
        self._auth_json_path = auth_json_path

    @property
    def install_template_path(self) -> Path:
        return Path(__file__).parent / "install-codex.sh"

    @staticmethod
    def _normalize_auth_json(raw: str, *, source: str) -> str:
        text = (raw or "").strip()
        if not text:
            raise ValueError(f"{source} is empty")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source} must be valid JSON") from exc
        return json.dumps(parsed, ensure_ascii=False)

    def _resolve_auth_json(self) -> tuple[str, str | None]:
        if self._auth_json and self._auth_json.strip():
            return self._normalize_auth_json(self._auth_json, source="auth_json"), None

        env_auth_json = self._env_get("CODEX_AUTH_JSON", "")
        if env_auth_json and env_auth_json.strip():
            return self._normalize_auth_json(env_auth_json, source="CODEX_AUTH_JSON"), None

        auth_json_path = self._auth_json_path or self._env_get("CODEX_AUTH_JSON_PATH", "")
        if auth_json_path and auth_json_path.strip():
            raw = Path(auth_json_path).expanduser().read_text(encoding="utf-8")
            return self._normalize_auth_json(raw, source="CODEX_AUTH_JSON_PATH"), None

        api_key = (self._env_get("OPENAI_API_KEY", "") or "").strip()
        if api_key:
            return json.dumps({"OPENAI_API_KEY": api_key}, ensure_ascii=False), api_key

        raise ValueError(
            "codex adapter requires either OPENAI_API_KEY or a valid auth.json via "
            "CODEX_AUTH_JSON/CODEX_AUTH_JSON_PATH/auth_json"
        )

    def create_run_commands(self, instruction: str, *, task, seed: int) -> list[ExecCommand]:
        escaped_instruction = shlex.quote(instruction)
        if not self.model_name:
            raise ValueError("model_name is required")
        model = self.model_name.split("/")[-1]
        auth_json, api_key = self._resolve_auth_json()
        env = {
            "CODEX_HOME": "/tmp/codex-home",
            "CODEX_AUTH_JSON": auth_json,
        }
        if api_key:
            env["OPENAI_API_KEY"] = api_key
            env["CODEX_API_KEY"] = api_key
        reasoning_effort = self._reasoning_effort
        reasoning_flag = f"-c model_reasoning_effort={reasoning_effort} " if reasoning_effort else ""
        setup_cmd = """python3 - <<'PY'
from pathlib import Path
import os

codex_home = Path(os.environ["CODEX_HOME"])
codex_home.mkdir(parents=True, exist_ok=True)
auth_json = os.environ.get("CODEX_AUTH_JSON", "").strip()
if not auth_json:
    raise SystemExit("CODEX_AUTH_JSON is required")
(codex_home / "auth.json").write_text(auth_json + "\\n", encoding="utf-8")
PY"""
        return [
            ExecCommand(
                cmd=setup_cmd,
                env=env,
            ),
            ExecCommand(
                cmd=(
                    "codex exec "
                    "--dangerously-bypass-approvals-and-sandbox "
                    "--skip-git-repo-check "
                    f"--model {model} "
                    "--json "
                    "--enable unified_exec "
                    f"{reasoning_flag}"
                    "-- "
                    f"{escaped_instruction} "
                    "2>&1 </dev/null | tee /logs/agent/codex.txt; "
                    'mkdir -p /logs/agent/sessions; '
                    'if [ -d "$CODEX_HOME/sessions" ]; then '
                    'cp -R "$CODEX_HOME/sessions/." /logs/agent/sessions/ 2>/dev/null || true; '
                    "fi"
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
        parsed = parse_codex_session(agent_logs, instruction=instruction)
        if parsed is None and results:
            parsed = parse_codex_exec_output(results[-1].stdout or "", instruction=instruction)
        if parsed is None:
            return await super().parse_run_artifacts(
                task=task, instruction=instruction, results=results, artifacts=artifacts
            )
        return self._events_from_parse_result(
            artifacts,
            parsed,
            parser_name="codex-session-or-json",
            raw_artifacts=["agent/codex.txt", "agent/sessions/**/*.jsonl"],
        )
