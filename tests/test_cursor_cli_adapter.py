from sentient_evals.adapters.installed.cursor_cli import CursorCliAdapter
from sentient_evals.models import Task


def test_cursor_cli_command_enables_non_interactive_flags(monkeypatch):
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    adapter = CursorCliAdapter(model_name="cursor/auto")

    commands = adapter.create_run_commands("fix issue", task=Task(id="t1"), seed=1)
    assert len(commands) == 1
    cmd = commands[0].cmd

    assert "--print" in cmd
    assert "--output-format text" in cmd
    assert "AGENT_HELP" in cmd
    assert "--force" in cmd
    assert "--trust" in cmd
    assert "--yolo" in cmd
    assert "${EXTRA_FLAGS}" in cmd


def test_cursor_cli_requires_api_key(monkeypatch):
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    adapter = CursorCliAdapter(model_name="cursor/auto")

    try:
        adapter.create_run_commands("fix issue", task=Task(id="t1"), seed=1)
    except ValueError as exc:
        assert "CURSOR_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected ValueError when CURSOR_API_KEY is missing")
