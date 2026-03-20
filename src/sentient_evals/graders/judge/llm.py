from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Sequence

from ...artifacts import TrialArtifacts
from ...judges import JudgeClient, LiteLLMJudgeClient
from ...models import GraderResult, Severity, Task, TranscriptEvent


@dataclass(frozen=True)
class LLMJudgeConfig:
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 512
    temperature: float = 0.0


@dataclass(frozen=True)
class LLMJudgeGrader:
    name: str = "llm_judge"
    rubric: str = "Decide whether the model answer satisfies the task."
    reference: str | None = None
    answer_field: str = "answer"
    config: LLMJudgeConfig | None = None
    client: JudgeClient | None = None

    @staticmethod
    def _judge_task_input(task: Task, transcript: Sequence[TranscriptEvent]) -> str:
        if task.input:
            return str(task.input)
        for event in transcript:
            if event.kind == "message" and event.role == "user" and (event.content or "").strip():
                return str(event.content)
        return ""

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

        answer = ""
        if isinstance(outcome, dict):
            answer = str(outcome.get(self.answer_field, ""))
        else:
            answer = str(getattr(outcome, self.answer_field, "") if outcome is not None else "")
        task_input = self._judge_task_input(task, transcript)

        prompt = (
            "You are a strict evaluator.\n\n"
            f"RUBRIC:\n{self.rubric}\n\n"
            f"TASK INPUT:\n{task_input}\n\n"
            + (f"REFERENCE:\n{self.reference}\n\n" if self.reference else "")
            + f"MODEL ANSWER:\n{answer}\n\n"
            "Return JSON with keys: verdict (pass|fail|unknown), score (0-1), reason (string)."
        )
        artifacts.judge().write_text("prompt.txt", prompt)

        client = self.client
        if client is None:
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
            client = LiteLLMJudgeClient(api_key=api_key)
        try:
            resp = await client.score(
                model=self.config.model,
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except Exception as exc:
            artifacts.judge().write_json("error.json", {"error": str(exc)})
            return GraderResult(
                name=self.name,
                score=0.0,
                passed=False,
                severity=Severity.error,
                details={"error": str(exc)},
            )

        artifacts.judge().write_json("response.json", resp.raw)

        verdict = "unknown"
        score = 0.0
        reason = ""
        try:
            payload = json.loads(resp.content)
            verdict = str(payload.get("verdict", "unknown")).lower()
            score = float(payload.get("score", 0.0))
            reason = str(payload.get("reason", ""))
        except Exception:
            verdict = "unknown"
            score = 0.0
            reason = "Failed to parse judge response"

        passed = verdict == "pass"
        artifacts.judge().write_json(
            "verdict.json",
            {
                "verdict": verdict,
                "passed": passed,
                "score": score,
                "reason": reason,
                "judge_model": self.config.model,
            },
        )
        severity = Severity.info if passed else Severity.error
        if verdict == "unknown":
            severity = Severity.warning
        return GraderResult(
            name=self.name,
            score=score,
            passed=passed,
            severity=severity,
            details={
                "verdict": verdict,
                "reason": reason,
                "judge_model": self.config.model,
            },
        )
