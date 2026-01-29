import asyncio
from pathlib import Path

from sentient_evals.agent_file import load_agent_adapter_from_file, parse_agent_file_ref
from sentient_evals.env import LocalToolExecutor
from sentient_evals.models import Task


def test_parse_agent_file_ref(tmp_path: Path):
    p = tmp_path / "a.py"
    p.write_text("x = 1\n", encoding="utf-8")
    ref = parse_agent_file_ref(f"{p}:agent")
    assert ref.path == p.resolve()
    assert ref.attr == "agent"


def test_load_agent_callable_wrapped(tmp_path: Path):
    p = tmp_path / "agent.py"
    p.write_text(
        "\n".join(
            [
                "async def my_agent(input: str, tools, seed: int):",
                "    return {'answer': 'ok'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ref = parse_agent_file_ref(f"{p}:my_agent")
    adapter = load_agent_adapter_from_file(ref)
    env = LocalToolExecutor(root=tmp_path)
    transcript, outcome = asyncio.run(adapter.run(Task(id="t1", input={"input": "hi"}), seed=1, env=env))
    assert outcome.data.get("answer") == "ok"
    assert transcript


def test_load_agent_factory_returns_adapter(tmp_path: Path):
    p = tmp_path / "agent.py"
    p.write_text(
        "\n".join(
            [
                "class A:",
                "    name = 'a'",
                "    async def run(self, task, *, seed: int, env):",
                "        from sentient_evals.models import Outcome, TranscriptEvent",
                "        return ([TranscriptEvent(kind='message', role='user', content='x')], Outcome(summary='ok', data={'answer': 'x'}))",
                "",
                "def build():",
                "    return A()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ref = parse_agent_file_ref(f"{p}:build")
    adapter = load_agent_adapter_from_file(ref)
    env = LocalToolExecutor(root=tmp_path)
    transcript, outcome = asyncio.run(adapter.run(Task(id="t1"), seed=1, env=env))
    assert outcome.data.get("answer") == "x"
    assert transcript
