from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from ...artifacts import TrialArtifacts
from ...models import GraderResult, Severity, Task, TranscriptEvent


def _args_match(expected: dict[str, Any] | None, actual: dict[str, Any]) -> bool:
    if not expected:
        return True
    for key, val in expected.items():
        if isinstance(val, dict) and "$regex" in val:
            if re.search(str(val["$regex"]), str(actual.get(key, ""))) is None:
                return False
        else:
            if actual.get(key) != val:
                return False
    return True


@dataclass(frozen=True)
class ToolUsageRule:
    tool: str
    required: bool = True
    args: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolUsageGrader:
    name: str = "tool_usage"
    required_tools: Sequence[str] = ()
    forbidden_tools: Sequence[str] = ()
    rules: Sequence[ToolUsageRule] = ()
    max_turns: int | None = None
    max_tool_calls: int | None = None

    async def grade(
        self, *, task: Task, transcript: Sequence[TranscriptEvent], outcome, artifacts: TrialArtifacts
    ) -> GraderResult:
        tool_calls = [e.tool_call for e in transcript if e.tool_call is not None]
        tool_names = [c.name for c in tool_calls]

        missing = [t for t in self.required_tools if t not in tool_names]
        forbidden = [t for t in self.forbidden_tools if t in tool_names]

        rule_failures: list[dict[str, Any]] = []
        for rule in self.rules:
            matched = [
                c
                for c in tool_calls
                if c.name == rule.tool and _args_match(rule.args, c.args or {})
            ]
            if rule.required and not matched:
                rule_failures.append({"tool": rule.tool, "required": True, "args": rule.args})
            if not rule.required and matched:
                rule_failures.append({"tool": rule.tool, "required": False, "args": rule.args})

        turn_count = len([e for e in transcript if e.kind == "message"])
        tool_call_count = len(tool_calls)

        limit_failures = []
        if self.max_turns is not None and turn_count > self.max_turns:
            limit_failures.append({"max_turns": self.max_turns, "actual": turn_count})
        if self.max_tool_calls is not None and tool_call_count > self.max_tool_calls:
            limit_failures.append({"max_tool_calls": self.max_tool_calls, "actual": tool_call_count})

        passed = not (missing or forbidden or rule_failures or limit_failures)
        artifacts.verifier().write_json(
            "tool_usage.json",
            {
                "missing": missing,
                "forbidden": forbidden,
                "rule_failures": rule_failures,
                "limits": limit_failures,
                "turn_count": turn_count,
                "tool_call_count": tool_call_count,
            },
        )
        return GraderResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            severity=Severity.info if passed else Severity.error,
            details={
                "missing": missing,
                "forbidden": forbidden,
                "rule_failures": rule_failures,
                "limits": limit_failures,
            },
        )
