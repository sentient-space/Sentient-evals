"""Pin the UI -> harness grader-name contract.

The Sentient web playground emits these `type` strings when authoring graders
(see `playground-adapters.ts::evaluatorTypeFromKind`). Every entry below must
build successfully via `sentient_evals.registry.build_grader`. Adding or
renaming a UI grader kind without keeping this list in sync is the bug class
this test exists to catch.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from sentient_evals.artifacts import TrialArtifacts
from sentient_evals.graders import (
    LLMJudgeGrader,
    StateCheckGrader,
    StaticAnalysisGrader,
    ToolUsageGrader,
    TranscriptGrader,
    VerifierScriptGrader,
)
from sentient_evals.models import Task, ToolCall, TranscriptEvent
from sentient_evals.registry import build_grader


# Sentient web `evaluatorTypeFromKind` mapping. Keep in lockstep with
# packages/web/src/lib/evals/playground-adapters.ts.
UI_GRADER_KIND_TO_TYPE: dict[str, str] = {
    "deterministic": "verifier_script",
    "state_check": "state_check",
    "tool_calls": "tool_calls",
    "llm_rubric": "llm_judge",
    "static_analysis": "static_analysis",
    "transcript": "transcript",
}


# Minimal config for each backend type that produces a usable grader.
MINIMAL_CONFIG: dict[str, dict] = {
    "verifier_script": {},
    "state_check": {},
    "tool_calls": {},
    "tool_usage": {},
    "llm_judge": {"rubric": "Decide whether the model answer satisfies the task."},
    "llm_rubric": {"rubric": "Decide whether the model answer satisfies the task."},
    "static_analysis": {},
    "transcript": {"must_contain": []},
}


EXPECTED_TYPE: dict[str, type] = {
    "verifier_script": VerifierScriptGrader,
    "state_check": StateCheckGrader,
    "tool_calls": ToolUsageGrader,
    "tool_usage": ToolUsageGrader,
    "llm_judge": LLMJudgeGrader,
    "llm_rubric": LLMJudgeGrader,
    "static_analysis": StaticAnalysisGrader,
    "transcript": TranscriptGrader,
}


@pytest.mark.parametrize("ui_kind,backend_type", sorted(UI_GRADER_KIND_TO_TYPE.items()))
def test_every_ui_grader_kind_maps_to_a_registered_type(ui_kind: str, backend_type: str) -> None:
    spec = {"type": backend_type, "config": MINIMAL_CONFIG[backend_type]}
    grader = build_grader(spec)
    assert isinstance(grader, EXPECTED_TYPE[backend_type]), (
        f"UI kind '{ui_kind}' (-> '{backend_type}') built {type(grader).__name__}, "
        f"expected {EXPECTED_TYPE[backend_type].__name__}"
    )


def test_tool_calls_alias_resolves_to_tool_usage() -> None:
    grader = build_grader({"type": "tool_calls", "config": {"required_tools": ["exec"]}})
    assert isinstance(grader, ToolUsageGrader)
    assert grader.required_tools == ["exec"]


def test_llm_rubric_alias_resolves_to_llm_judge() -> None:
    grader = build_grader({"type": "llm_rubric", "config": {"rubric": "Be helpful."}})
    assert isinstance(grader, LLMJudgeGrader)
    assert grader.rubric == "Be helpful."


def test_transcript_grader_pass_when_required_phrase_present() -> None:
    grader = build_grader(
        {"type": "transcript", "config": {"must_contain": ["sorry"], "roles": ["assistant"]}}
    )
    transcript = [
        TranscriptEvent(kind="message", role="assistant", content="I'm sorry about that."),
    ]
    result = asyncio.run(
        grader.grade(task=Task(id="t1"), transcript=transcript, outcome={}, artifacts=TrialArtifacts(Path(".")))
    )
    assert result.passed is True


def test_transcript_grader_fail_when_forbidden_phrase_present() -> None:
    grader = build_grader(
        {
            "type": "transcript",
            "config": {"must_not_contain": ["secret_token"], "case_sensitive": True},
        }
    )
    transcript = [
        TranscriptEvent(kind="message", role="assistant", content="here is your secret_token"),
    ]
    result = asyncio.run(
        grader.grade(task=Task(id="t1"), transcript=transcript, outcome={}, artifacts=TrialArtifacts(Path(".")))
    )
    assert result.passed is False
    assert result.details["forbidden_hit"] == ["secret_token"]


def test_transcript_grader_regex_mode() -> None:
    grader = build_grader(
        {"type": "transcript", "config": {"must_contain": [r"^\d+\.\s"], "regex": True}}
    )
    transcript = [
        TranscriptEvent(kind="message", role="assistant", content="1. first item\n2. second item"),
    ]
    result = asyncio.run(
        grader.grade(task=Task(id="t1"), transcript=transcript, outcome={}, artifacts=TrialArtifacts(Path(".")))
    )
    assert result.passed is True


def test_unknown_grader_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown grader type"):
        build_grader({"type": "this_does_not_exist"})


def test_transcript_grader_validates_must_contain_type() -> None:
    with pytest.raises(ValueError, match="must_contain"):
        build_grader({"type": "transcript", "config": {"must_contain": "not a list"}})


def _ignored_unused_imports() -> None:
    # Keep imports tidy without `# noqa` clutter at module top.
    _ = ToolCall(name="x")
