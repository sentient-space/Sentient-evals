import asyncio
from pathlib import Path

from sentient_evals.artifacts import TrialArtifacts
from sentient_evals.env import LocalToolExecutor
from sentient_evals.graders import (
    BudgetGrader,
    StateCheckGrader,
    StaticAnalysisGrader,
    StaticAnalysisSpec,
    ToolUsageGrader,
    ToolUsageRule,
    VerifierScriptGrader,
    VerifierScriptSpec,
)
from sentient_evals.models import Task, ToolCall, TranscriptEvent


def test_state_check_grader_passes():
    grader = StateCheckGrader(expect={"answer": {"$eq": "ok"}})
    task = Task(id="t1")
    result = asyncio.run(
        grader.grade(task=task, transcript=[], outcome={"answer": "ok"}, artifacts=TrialArtifacts(Path(".")))
    )
    assert result.passed is True


def test_tool_usage_grader_required_and_forbidden():
    grader = ToolUsageGrader(required_tools=["exec"], forbidden_tools=["rm"])
    transcript = [TranscriptEvent(kind="tool_call", tool_call=ToolCall(name="exec", args={}))]
    result = asyncio.run(
        grader.grade(task=Task(id="t1"), transcript=transcript, outcome={}, artifacts=TrialArtifacts(Path(".")))
    )
    assert result.passed is True


def test_tool_usage_grader_rule_failure():
    grader = ToolUsageGrader(rules=[ToolUsageRule(tool="read_file", required=True, args={"path": "a.txt"})])
    transcript = [
        TranscriptEvent(kind="tool_call", tool_call=ToolCall(name="read_file", args={"path": "b.txt"}))
    ]
    result = asyncio.run(
        grader.grade(task=Task(id="t1"), transcript=transcript, outcome={}, artifacts=TrialArtifacts(Path(".")))
    )
    assert result.passed is False


def test_static_analysis_grader_runs_commands(tmp_path: Path):
    grader = StaticAnalysisGrader(checks=[StaticAnalysisSpec(name="ok", cmd="python -c 'print(1)'")])
    env = LocalToolExecutor(root=tmp_path)
    result = asyncio.run(
        grader.grade_with_env(
            task=Task(id="t1"), transcript=[], outcome={}, artifacts=TrialArtifacts(tmp_path), env=env
        )
    )
    assert result.passed is True


def test_verifier_script_grader_reads_reward(tmp_path: Path):
    env = LocalToolExecutor(root=tmp_path)
    spec = VerifierScriptSpec(
        cmd="sh -lc 'mkdir -p logs/verifier && echo 1 > logs/verifier/reward.txt'",
        reward_paths=("logs/verifier/reward.txt",),
    )
    grader = VerifierScriptGrader(spec=spec)
    result = asyncio.run(
        grader.grade_with_env(
            task=Task(id="t1"), transcript=[], outcome={}, artifacts=TrialArtifacts(tmp_path), env=env
        )
    )
    assert result.passed is True


def test_budget_grader_detects_missing_metrics():
    grader = BudgetGrader(max_tokens=10)
    result = asyncio.run(
        grader.grade(task=Task(id="t1"), transcript=[], outcome={}, artifacts=TrialArtifacts(Path(".")))
    )
    assert result.passed is False
