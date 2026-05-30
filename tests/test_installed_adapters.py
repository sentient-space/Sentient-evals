import asyncio
import json
from pathlib import Path

from sentient_evals.adapters.installed.aider import AiderAdapter
from sentient_evals.adapters.installed.base import BaseInstalledAdapter, ExecCommand
from sentient_evals.adapters.installed.cline_cli import ClineCliAdapter
from sentient_evals.adapters.installed.codex import CodexAdapter
from sentient_evals.adapters.installed.cursor_cli import CursorCliAdapter
from sentient_evals.adapters.installed.gemini_cli import GeminiCliAdapter
from sentient_evals.adapters.installed.goose import GooseAdapter
from sentient_evals.adapters.installed.opencode import OpenCodeAdapter
from sentient_evals.adapters.installed.qwen_code import QwenCodeAdapter
from sentient_evals.artifacts import TrialArtifacts
from sentient_evals.env import ExecResult
from sentient_evals.models import Outcome, Task, TranscriptEvent
from sentient_evals.replay import RecordingToolExecutor


class FakeToolExecutor:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.commands: list[str] = []
        self.sync_logs_calls = 0

    async def exec(self, cmd: str, *, timeout_s: float | None = None) -> ExecResult:
        self.commands.append(cmd)
        return ExecResult(stdout=f"ran:{cmd}", stderr="", exit_code=0, duration_ms=1)

    async def read_file(self, path: str) -> str:
        return self.files.get(path, "")

    async def write_file(self, path: str, content: str) -> None:
        self.files[path] = content

    async def list_dir(self, path: str) -> list[str]:
        return []

    async def call(self, tool_name: str, args: dict) -> dict:
        return {"tool": tool_name, "args": args}

    async def sync_logs(self) -> None:
        self.sync_logs_calls += 1


class DummyInstalledAdapter(BaseInstalledAdapter):
    name = "dummy-installed"

    def __init__(self, template_path: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._template_path = template_path

    @property
    def install_template_path(self) -> Path:
        return self._template_path

    def create_run_commands(self, instruction: str, *, task: Task, seed: int):
        return [ExecCommand(cmd=f"echo {instruction}")]


class DummyParsingAdapter(DummyInstalledAdapter):
    async def parse_run_artifacts(self, *, task, instruction, results, artifacts):
        return [
            TranscriptEvent(kind="message", role="user", content=instruction),
            TranscriptEvent(kind="message", role="assistant", content="done"),
        ]


def test_install_template_rendering(tmp_path: Path):
    tmpl = tmp_path / "install.sh"
    tmpl.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                "{% if version %}",
                "echo version={{ version }}",
                "{% else %}",
                "echo version=latest",
                "{% endif %}",
            ]
        ),
        encoding="utf-8",
    )
    adapter = DummyInstalledAdapter(template_path=tmpl, version="1.2.3")
    env = FakeToolExecutor()
    artifacts = TrialArtifacts(tmp_path)
    asyncio.run(adapter.install(env, artifacts))
    rendered = (tmp_path / "agent" / "install.sh").read_text(encoding="utf-8")
    assert "version=1.2.3" in rendered


def test_installed_adapter_run_writes_artifacts(tmp_path: Path):
    tmpl = tmp_path / "install.sh"
    tmpl.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    adapter = DummyInstalledAdapter(template_path=tmpl)
    env = FakeToolExecutor()
    artifacts = TrialArtifacts(tmp_path)
    transcript, outcome = asyncio.run(
        adapter.run(task=Task(id="t1"), instruction="hello", seed=1, env=env, artifacts=artifacts)
    )
    assert isinstance(outcome, Outcome)
    assert isinstance(transcript, list)
    assert (tmp_path / "agent" / "install.sh").exists()
    assert (tmp_path / "agent" / "commands" / "0" / "stdout.txt").exists()
    diagnostics = json.loads((tmp_path / "agent" / "parsed" / "diagnostics.json").read_text())
    assert diagnostics["parser_status"] == "fallback_stdout"
    assert diagnostics["capabilities"]["tool_call_support"] == "none"


def test_installed_adapter_env_overrides_do_not_require_process_env(tmp_path: Path):
    adapter = AiderAdapter(model_name="openai/gpt-4o-mini", env_overrides={"OPENAI_API_KEY": "test-key"})
    commands = adapter.create_run_commands("solve it", task=Task(id="t1"), seed=1)
    assert commands
    assert commands[0].env == {"AIDER_API_KEY": "openai=test-key"}


def test_cursor_cli_uses_stream_json_output(tmp_path: Path):
    del tmp_path
    adapter = CursorCliAdapter(model_name="cursor/auto", env_overrides={"CURSOR_API_KEY": "cursor-test-key"})
    commands = adapter.create_run_commands("solve it", task=Task(id="t1"), seed=1)
    assert commands
    assert "--output-format stream-json" in commands[0].cmd
    assert commands[0].env == {"CURSOR_API_KEY": "cursor-test-key"}


def test_weak_adapters_use_structured_output_flags(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    gemini = GeminiCliAdapter(model_name="google/gemini-2.5-pro")
    assert "--output-format stream-json" in gemini.create_run_commands("solve", task=Task(id="t1"), seed=1)[0].cmd

    qwen = QwenCodeAdapter(model_name="qwen3-coder-plus")
    assert "--output-format stream-json" in qwen.create_run_commands("solve", task=Task(id="t1"), seed=1)[0].cmd

    cline = ClineCliAdapter(model_name="anthropic:claude-sonnet-4-5")
    cline_commands = cline.create_run_commands("solve", task=Task(id="t1"), seed=1)
    assert "--json" in cline_commands[1].cmd
    assert "-F plain" not in cline_commands[1].cmd

    opencode = OpenCodeAdapter(model_name="anthropic/claude-sonnet-4-5")
    assert "--format=json" in opencode.create_run_commands("solve", task=Task(id="t1"), seed=1)[0].cmd

    goose = GooseAdapter(model_name="anthropic/claude-sonnet-4-5")
    goose_commands = goose.create_run_commands("solve", task=Task(id="t1"), seed=1)
    assert "--output-format stream-json" in goose_commands[1].cmd


def test_codex_uses_tmp_home_and_exports_sessions(tmp_path: Path):
    del tmp_path
    adapter = CodexAdapter(model_name="openai/o4-mini", env_overrides={"OPENAI_API_KEY": "test-key"})
    commands = adapter.create_run_commands("solve it", task=Task(id="t1"), seed=1)
    assert len(commands) == 2
    assert commands[0].env is not None
    assert commands[0].env["CODEX_HOME"] == "/tmp/codex-home"
    assert commands[1].env is not None
    assert commands[1].env["CODEX_HOME"] == "/tmp/codex-home"
    assert 'tee /logs/agent/codex.txt; ' in commands[1].cmd
    assert 'cp -R "$CODEX_HOME/sessions/." /logs/agent/sessions/' in commands[1].cmd


def test_installed_adapter_run_syncs_logs_before_parsing(tmp_path: Path):
    tmpl = tmp_path / "install.sh"
    tmpl.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    adapter = DummyParsingAdapter(template_path=tmpl)
    env = FakeToolExecutor()
    artifacts = TrialArtifacts(tmp_path)

    transcript, outcome = asyncio.run(
        adapter.run(task=Task(id="t1"), instruction="hello", seed=1, env=env, artifacts=artifacts)
    )

    assert env.sync_logs_calls == 1
    assert isinstance(outcome, Outcome)
    assert len(transcript) == 2


def test_installed_adapter_run_syncs_logs_through_recording_executor(tmp_path: Path):
    tmpl = tmp_path / "install.sh"
    tmpl.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    adapter = DummyParsingAdapter(template_path=tmpl)
    inner = FakeToolExecutor()
    env = RecordingToolExecutor(inner, log_path=tmp_path / "replay" / "tool_calls.jsonl")
    artifacts = TrialArtifacts(tmp_path)

    try:
        transcript, outcome = asyncio.run(
            adapter.run(task=Task(id="t1"), instruction="hello", seed=1, env=env, artifacts=artifacts)
        )
    finally:
        env.close()

    assert inner.sync_logs_calls == 1
    assert isinstance(outcome, Outcome)
    assert len(transcript) == 2
