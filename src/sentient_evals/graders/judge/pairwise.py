from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Sequence

from ...artifacts import TrialArtifacts
from ...judges import JudgeClient, LiteLLMJudgeClient
from ...models import GraderResult, Severity, Task, TranscriptEvent


@dataclass(frozen=True)
class PairwiseJudgeConfig:
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 512
    temperature: float = 0.0


@dataclass(frozen=True)
class PairwiseJudgeGrader:
    name: str = "pairwise_judge"
    rubric: str = "Compare the two responses and pick the better one."
    config: PairwiseJudgeConfig | None = None
    client: JudgeClient | None = None
    candidate_key: str = "candidate"
    baseline_key: str = "baseline"

    async def grade(
        self, *, task: Task, transcript: Sequence[TranscriptEvent], outcome: Any, artifacts: TrialArtifacts
    ) -> GraderResult:
        if self.config is None:
            return GraderResult(
                name=self.name,
                score=0.0,
                passed=False,
                severity=Severity.error,
                details={"error": "PairwiseJudgeGrader requires config"},
            )

        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            return GraderResult(
                name=self.name,
                score=0.0,
                passed=False,
                severity=Severity.error,
                details={"error": f"Missing API key env var: {self.config.api_key_env}"},
            )

        if not isinstance(outcome, dict):
            return GraderResult(
                name=self.name,
                score=0.0,
                passed=False,
                severity=Severity.error,
                details={"error": "PairwiseJudgeGrader requires outcome dict with candidate/baseline"},
            )

        candidate = str(outcome.get(self.candidate_key, ""))
        baseline = str(outcome.get(self.baseline_key, ""))

        prompt = (
            "You are a strict evaluator.\n\n"
            f"RUBRIC:\n{self.rubric}\n\n"
            f"TASK INPUT:\n{task.input}\n\n"
            f"CANDIDATE RESPONSE:\n{candidate}\n\n"
            f"BASELINE RESPONSE:\n{baseline}\n\n"
            "Return JSON with keys: verdict (candidate|baseline|tie|unknown), score (0-1), reason (string)."
        )
        artifacts.judge().write_text("prompt.txt", prompt)

        client = self.client or LiteLLMJudgeClient()
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
        artifacts.judge().write_text("response.txt", resp.content)

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

        passed = verdict == "candidate"
        severity = Severity.info if passed else Severity.error
        if verdict in {"tie", "unknown"}:
            severity = Severity.warning

        artifacts.judge().write_json(
            "verdict.json",
            {"verdict": verdict, "score": score, "reason": reason, "judge_model": self.config.model},
        )
        return GraderResult(
            name=self.name,
            score=score,
            passed=passed,
            severity=severity,
            details={"verdict": verdict, "reason": reason, "judge_model": self.config.model},
        )
