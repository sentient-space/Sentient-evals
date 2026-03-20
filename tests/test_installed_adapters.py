import asyncio
from pathlib import Path

from sentient_evals.adapters.installed.aider import AiderAdapter
from sentient_evals.adapters.installed.base import BaseInstalledAdapter, ExecCommand
from sentient_evals.adapters.installed.codex import CodexAdapter
from sentient_evals.artifacts import TrialArtifacts
from sentient_evals.env import ExecResult
from sentient_evals.models import Outcome, Task


class FakeToolExecutor:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.commands: list[str] = []

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


def test_installed_adapter_env_overrides_do_not_require_process_env(tmp_path: Path):
    adapter = AiderAdapter(model_name="openai/gpt-4o-mini", env_overrides={"OPENAI_API_KEY": "test-key"})
    commands = adapter.create_run_commands("solve it", task=Task(id="t1"), seed=1)
    assert commands
    assert commands[0].env == {"AIDER_API_KEY": "openai=test-key"}
