from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .artifacts import TrialArtifacts
from .models import GraderResult, Severity, Task, TranscriptEvent


class Grader(Protocol):
    name: str

    async def grade(
        self,
        *,
        task: Task,
        transcript: Sequence[TranscriptEvent],
        outcome: Any,
        artifacts: TrialArtifacts,
    ) -> GraderResult:
        ...


@dataclass(frozen=True)
class ExactMatchGrader:
    name: str = "exact_match"
    field: str = "answer"
    expected: str = ""

    async def grade(
        self, *, task: Task, transcript: Sequence[TranscriptEvent], outcome: Any, artifacts: TrialArtifacts
    ) -> GraderResult:
        actual = ""
        if isinstance(outcome, dict):
            actual = str(outcome.get(self.field, ""))
        else:
            actual = str(getattr(outcome, self.field, "") if outcome is not None else "")
        passed = actual == self.expected
        return GraderResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            severity=Severity.info if passed else Severity.error,
            details={"expected": self.expected, "actual": actual, "field": self.field},
        )


@dataclass(frozen=True)
class LLMJudgeConfig:
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 512
    temperature: float = 0.0


@dataclass(frozen=True)
class LLMJudgeGrader:
    """
    Minimal LLM-as-judge grader scaffold.

    This is intentionally thin: production setups should add calibration sets,
    multi-judge voting, and drift checks, as recommended in Anthropic's evals guide:
    https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
    """

    name: str = "llm_judge"
    rubric: str = "Decide whether the model answer satisfies the task. Return PASS or FAIL."
    reference: str | None = None
    answer_field: str = "answer"
    config: LLMJudgeConfig | None = None

    async def grade(
        self, *, task: Task, transcript: Sequence[TranscriptEvent], outcome: Any, artifacts: TrialArtifacts
    ) -> GraderResult:
        if self.config is None:
            artifacts.judge().write_json("error.json", {"error": "LLMJudgeGrader requires config"})
            return GraderResult(
                name=self.name,
                score=0.0,
                passed=False,
                severity=Severity.error,
                details={"error": "LLMJudgeGrader requires config"},
            )

        try:
            from litellm import acompletion  # type: ignore
        except Exception as e:  # pragma: no cover
            artifacts.judge().write_json(
                "error.json",
                {
                    "error": "Install sentient-evals[llm] to use LLMJudgeGrader",
                    "import_error": str(e),
                },
            )
            return GraderResult(
                name=self.name,
                score=0.0,
                passed=False,
                severity=Severity.error,
                details={"error": "Install sentient-evals[llm] to use LLMJudgeGrader", "import_error": str(e)},
            )

        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            artifacts.judge().write_json(
                "error.json", {"error": f"Missing API key env var: {self.config.api_key_env}"}
            )
            return GraderResult(
                name=self.name,
                score=0.0,
                passed=False,
                severity=Severity.error,
                details={"error": f"Missing API key env var: {self.config.api_key_env}"},
            )

        answer = ""
        if isinstance(outcome, dict):
            answer = str(outcome.get(self.answer_field, ""))
        else:
            answer = str(getattr(outcome, self.answer_field, "") if outcome is not None else "")

        prompt = (
            "You are a strict evaluator.\n\n"
            f"RUBRIC:\n{self.rubric}\n\n"
            f"TASK INPUT:\n{task.input}\n\n"
            + (f"REFERENCE:\n{self.reference}\n\n" if self.reference else "")
            + f"MODEL ANSWER:\n{answer}\n\n"
            "Return exactly one token: PASS or FAIL."
        )
        artifacts.judge().write_text("prompt.txt", prompt)

        resp = await acompletion(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        try:
            raw = resp.model_dump()
        except Exception:  # pragma: no cover
            raw = {"raw": str(resp)}
        artifacts.judge().write_json("response.json", raw)

        content = (resp.choices[0].message.content or "").strip()
        artifacts.judge().write_text("response.txt", content)

        normalized = content.upper()
        passed = normalized.startswith("PASS")
        artifacts.judge().write_json(
            "verdict.json",
            {
                "passed": passed,
                "score": 1.0 if passed else 0.0,
                "judge_model": self.config.model,
                "raw_head": normalized[:200],
            },
        )
        return GraderResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            severity=Severity.info if passed else Severity.error,
            details={"judge_model": self.config.model, "raw": normalized[:200]},
        )


@dataclass(frozen=True)
class MultiJudgeAggregation(str):
    majority = "majority"
    unanimous = "unanimous"
    average_score = "average_score"


@dataclass(frozen=True)
class MultiLLMJudgeGrader:
    """
    Runs multiple judges and aggregates. This is a scaffold; the core harness should
    also ship calibration tooling and disagreement reporting.
    """

    name: str = "multi_llm_judge"
    judges: Sequence[LLMJudgeGrader] = ()
    aggregation: MultiJudgeAggregation = MultiJudgeAggregation.majority

    async def grade(
        self, *, task: Task, transcript: Sequence[TranscriptEvent], outcome: Any, artifacts: TrialArtifacts
    ) -> GraderResult:
        if not self.judges:
            artifacts.judge().write_json("error.json", {"error": "No judges configured"})
            return GraderResult(
                name=self.name,
                score=0.0,
                passed=False,
                severity=Severity.error,
                details={"error": "No judges configured"},
            )

        results: list[GraderResult] = []
        for i, j in enumerate(self.judges):
            j_art = artifacts.judge().scoped(f"judge_{i}")
            results.append(await j.grade(task=task, transcript=transcript, outcome=outcome, artifacts=j_art))

        scores = [r.score for r in results]
        passes = [r.passed for r in results]

        if self.aggregation == MultiJudgeAggregation.unanimous:
            passed = all(passes)
            score = float(sum(scores)) / float(len(scores))
        elif self.aggregation == MultiJudgeAggregation.average_score:
            score = float(sum(scores)) / float(len(scores))
            passed = score >= 0.5
        else:
            passed = sum(1 for p in passes if p) > (len(passes) // 2)
            score = float(sum(scores)) / float(len(scores))

        artifacts.judge().write_json(
            "aggregation.json",
            {
                "aggregation": self.aggregation,
                "passed": passed,
                "score": score,
                "judges": [{"name": r.name, "passed": r.passed, "score": r.score} for r in results],
            },
        )
        return GraderResult(
            name=self.name,
            score=score,
            passed=passed,
            severity=Severity.info if passed else Severity.error,
            details={
                "aggregation": self.aggregation,
                "judges": [{"name": r.name, "passed": r.passed, "score": r.score} for r in results],
            },
        )

