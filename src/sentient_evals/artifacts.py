from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .atif.models import TrajectorySchema
from .models import Outcome, TranscriptEvent, TrialConfig, TrialResult


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


class ArtifactWriter:
    def __init__(self, jobs_dir: Path, run_id: str):
        self.jobs_dir = jobs_dir
        self.run_id = run_id

    @property
    def run_dir(self) -> Path:
        return self.jobs_dir / self.run_id

    def trial_dir(self, trial_id: str) -> Path:
        return self.run_dir / "trials" / trial_id

    def ensure_run_dirs(self) -> None:
        (self.run_dir / "trials").mkdir(parents=True, exist_ok=True)

    def ensure_trial_dirs(self, trial_id: str) -> None:
        d = self.trial_dir(trial_id)
        (d / "judge").mkdir(parents=True, exist_ok=True)
        (d / "verifier").mkdir(parents=True, exist_ok=True)
        (d / "agent").mkdir(parents=True, exist_ok=True)

    def write_trial_config(self, cfg: TrialConfig) -> Path:
        p = self.trial_dir(cfg.trial_id) / "config.json"
        _atomic_write_text(p, cfg.model_dump_json(indent=2))
        return p

    def write_transcript(self, trial_id: str, events: Iterable[TranscriptEvent]) -> Path:
        p = self.trial_dir(trial_id) / "transcript.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for e in events:
                f.write(e.model_dump_json())
                f.write("\n")
        tmp.replace(p)
        return p

    def write_trajectory(self, trial_id: str, trajectory: TrajectorySchema) -> Path:
        p = self.trial_dir(trial_id) / "trajectory.json"
        _atomic_write_text(p, trajectory.model_dump_json(indent=2))
        return p

    def write_outcome(self, trial_id: str, outcome: Outcome) -> Path:
        p = self.trial_dir(trial_id) / "outcome.json"
        _atomic_write_text(p, outcome.model_dump_json(indent=2))
        return p

    def write_result(self, trial_id: str, result: TrialResult) -> Path:
        p = self.trial_dir(trial_id) / "result.json"
        _atomic_write_text(p, result.model_dump_json(indent=2))
        return p

    def write_json(self, relative_path: str, payload: object) -> Path:
        p = self.run_dir / relative_path
        _atomic_write_text(p, json.dumps(payload, indent=2, default=str))
        return p


class TrialArtifacts:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def scoped(self, relative_dir: str) -> TrialArtifacts:
        return TrialArtifacts(self.base_dir / relative_dir)

    def judge(self) -> TrialArtifacts:
        return self.scoped("judge")

    def verifier(self) -> TrialArtifacts:
        return self.scoped("verifier")

    def agent(self) -> TrialArtifacts:
        return self.scoped("agent")

    def write_text(self, relative_path: str, text: str) -> Path:
        p = self.base_dir / relative_path
        _atomic_write_text(p, text)
        return p

    def write_json(self, relative_path: str, payload: Any) -> Path:
        p = self.base_dir / relative_path
        _atomic_write_text(p, json.dumps(payload, indent=2, default=str))
        return p

