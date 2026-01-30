from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from ...artifacts import TrialArtifacts
from ...env import ToolExecutor
from ...models import GraderResult, Severity, Task, TranscriptEvent


@dataclass(frozen=True)
class VerifierScriptSpec:
    cmd: str = "sh -lc ./tests/test.sh"
    timeout_s: float | None = None
    reward_paths: Sequence[str] = (
        "/logs/verifier/reward.json",
        "/logs/verifier/reward.txt",
        "logs/verifier/reward.json",
        "logs/verifier/reward.txt",
    )


@dataclass(frozen=True)
class VerifierScriptGrader:
    name: str = "verifier_script"
    spec: VerifierScriptSpec = VerifierScriptSpec()

    async def grade(
        self, *, task: Task, transcript: Sequence[TranscriptEvent], outcome, artifacts: TrialArtifacts
    ) -> GraderResult:
        return GraderResult(
            name=self.name,
            score=0.0,
            passed=False,
            severity=Severity.error,
            details={"error": "VerifierScriptGrader requires env; use grade_with_env"},
        )

    async def grade_with_env(
        self,
        *,
        task: Task,
        transcript: Sequence[TranscriptEvent],
        outcome,
        artifacts: TrialArtifacts,
        env: ToolExecutor,
    ) -> GraderResult:
        res = await env.exec(self.spec.cmd, timeout_s=self.spec.timeout_s)
        # If the script exists but is not executable, retry via sh.
        if res.exit_code == 126 and "permission denied" in (res.stderr or "").lower():
            res = await env.exec("sh -lc 'sh ./tests/test.sh'", timeout_s=self.spec.timeout_s)
        artifacts.verifier().write_json(
            "verifier_exec.json",
            {
                "cmd": self.spec.cmd,
                "exit_code": res.exit_code,
                "duration_ms": res.duration_ms,
                "stdout": res.stdout[-2000:],
                "stderr": res.stderr[-2000:],
            },
        )

        reward_value = None
        reward_raw = None
        reward_path_used = None
        for path in self.spec.reward_paths:
            try:
                payload = await env.read_file(path)
            except Exception:
                continue
            reward_path_used = path
            reward_raw = payload.strip()
            if path.endswith(".json"):
                try:
                    data = json.loads(payload)
                    reward_value = float(data.get("reward", data.get("score", data.get("value"))))
                except Exception:
                    reward_value = None
            else:
                try:
                    reward_value = float(reward_raw)
                except Exception:
                    reward_value = None
            break

        artifacts.verifier().write_json(
            "reward.json",
            {"reward": reward_value, "raw": reward_raw, "path": reward_path_used},
        )

        if reward_value is None:
            return GraderResult(
                name=self.name,
                score=0.0,
                passed=False,
                severity=Severity.error,
                details={"error": "reward not found or invalid", "path": reward_path_used},
            )

        passed = reward_value >= 1.0
        return GraderResult(
            name=self.name,
            score=reward_value,
            passed=passed,
            severity=Severity.info if passed else Severity.error,
            details={"reward": reward_value, "path": reward_path_used},
        )
