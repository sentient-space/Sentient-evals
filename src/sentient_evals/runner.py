from __future__ import annotations

import asyncio
import random
import traceback
import inspect
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from math import sqrt
from typing import Any, Callable, Literal, Sequence, TypeVar
from contextlib import asynccontextmanager

from .adapters import AgentAdapter
from .atif.converters import transcript_to_trajectory
from . import __version__
from .artifacts import ArtifactWriter, TrialArtifacts
from .env import EnvironmentToolExecutor, LocalToolExecutor, ToolExecutor
from .environments.base import EnvironmentConfig, EnvironmentType
from .environments.compatibility import collect_compatibility_issues
from .environments.factory import EnvironmentFactory
from .graders import Grader, supports_env_grading
from .models import (
    Outcome,
    TranscriptEvent,
    Severity,
    RunConfigFile,
    RunResult,
    RunStatus,
    RunSummary,
    SuiteConfig,
    Task,
    GraderResult,
    TrialConfig,
    TrialResult,
    utcnow,
)
from .provenance import runtime_provenance
from .replay import RecordingToolExecutor, ReplayMode, ReplayingToolExecutor
from .task_bundles.models import TaskBundle


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    suite: SuiteConfig
    jobs_dir: Path
    adapter_name: str
    mode: Literal["fresh", "resume"] = "fresh"
    replay_mode: ReplayMode = "off"
    strict_replay: bool = True
    env_type: EnvironmentType = EnvironmentType.local_python
    docker_image_tag_prefix: str = "sentient-evals"
    daytona_snapshot_template: str | None = None
    daytona_network_block_all: bool | None = None


def _wilson_ci95(passed: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = 1.96
    phat = passed / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z * sqrt((phat * (1 - phat) + (z * z) / (4 * n)) / n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)


def _trial_id(task_id: str, attempt: int) -> str:
    return f"{task_id}__{attempt}"


def _best_effort_instruction(task: Task) -> str:
    for key in ("instruction", "prompt", "query", "question", "text", "input"):
        val = task.input.get(key) if isinstance(task.input, dict) else None
        if isinstance(val, str) and val.strip():
            return val
    if isinstance(task.input, dict) and len(task.input) == 1:
        only = next(iter(task.input.values()))
        if isinstance(only, str) and only.strip():
            return only
    return json.dumps(task.input, ensure_ascii=False) if task.input else task.id


async def _call_adapter(
    *,
    adapter: AgentAdapter,
    task: Task,
    instruction: str | None,
    seed: int,
    env: ToolExecutor,
    artifacts: TrialArtifacts,
) -> tuple[list[TranscriptEvent], Outcome]:
    fn = getattr(adapter, "run")
    sig = inspect.signature(fn)
    kwargs: dict[str, object] = {"task": task, "seed": seed, "env": env}
    if "instruction" in sig.parameters:
        kwargs["instruction"] = instruction
    if "artifacts" in sig.parameters:
        kwargs["artifacts"] = artifacts
    return await fn(**kwargs)


def _ensure_run_config(
    writer: ArtifactWriter, cfg: RunConfig, extra_provenance: dict[str, object] | None
) -> RunConfigFile:
    path = writer.run_dir / "run_config.json"
    if cfg.mode != "fresh" and path.exists():
        return RunConfigFile.model_validate_json(path.read_text(encoding="utf-8"))

    provenance = runtime_provenance()
    if extra_provenance:
        provenance = provenance | extra_provenance
    run_cfg = RunConfigFile(
        run_id=cfg.run_id,
        suite=cfg.suite,
        adapter=cfg.adapter_name,
        started_at=utcnow(),
        harness_version=__version__,
        provenance=provenance,
    )
    writer.write_json("run_config.json", run_cfg.model_dump())
    return run_cfg


T = TypeVar("T")


def _seed_trials(items: Sequence[T], trials_per_task: int, seeds: list[int]) -> list[tuple[T, int, int]]:
    trials: list[tuple[T, int, int]] = []
    for item in items:
        for attempt in range(trials_per_task):
            seed = seeds[attempt % len(seeds)]
            trials.append((item, attempt, seed))
    return trials


def _emit_event(on_event: Callable[[str, dict[str, Any]], None] | None, name: str, **payload: Any) -> None:
    if on_event is None:
        return
    try:
        on_event(name, payload)
    except Exception:
        pass


def _resume_trials(
    writer: ArtifactWriter,
    trials: list[tuple[T, int, int]],
    trial_id_fn: Callable[[T, int], str],
) -> tuple[list[TrialResult], list[tuple[T, int, int]]]:
    results: list[TrialResult] = []
    remaining: list[tuple[T, int, int]] = []
    for item, attempt, seed in trials:
        tid = trial_id_fn(item, attempt)
        existing = writer.trial_dir(tid) / "trial_result.json"
        if existing.exists():
            results.append(TrialResult.model_validate_json(existing.read_text(encoding="utf-8")))
        else:
            remaining.append((item, attempt, seed))
    return results, remaining


def _compute_stats(
    results: Sequence[TrialResult], tasks: Sequence[Task]
) -> tuple[int, int, float | None, float, float, dict[str, dict[str, float]]]:
    passed = 0
    failed = 0
    scores: list[float] = []
    per_task: dict[str, dict[str, float]] = {}
    for r in results:
        if not r.ok:
            failed += 1
            continue
        trial_passed = bool(r.graders) and all(g.passed for g in r.graders)
        if trial_passed:
            passed += 1
        else:
            failed += 1
        if r.graders:
            scores.append(sum(g.score for g in r.graders) / len(r.graders))
        t = per_task.setdefault(r.task_id, {"trials": 0.0, "passed": 0.0})
        t["trials"] += 1.0
        t["passed"] += 1.0 if trial_passed else 0.0

    pass_at_k = 0.0
    pass_pow_k = 0.0
    for task in tasks:
        t = per_task.get(task.id, {"trials": 0.0, "passed": 0.0})
        k = int(t["trials"])
        s = int(t["passed"])
        if k > 0:
            pass_at_k += 1.0 if s > 0 else 0.0
            pass_pow_k += 1.0 if s == k else 0.0
    if tasks:
        pass_at_k /= float(len(tasks))
        pass_pow_k /= float(len(tasks))

    avg_score = (sum(scores) / len(scores)) if scores else None
    return passed, failed, avg_score, pass_at_k, pass_pow_k, per_task


def _score_variance(scores: Sequence[float]) -> tuple[float | None, float | None]:
    if not scores:
        return None, None
    mean = sum(scores) / float(len(scores))
    var = sum((s - mean) ** 2 for s in scores) / float(len(scores))
    return var, sqrt(var)


def _collect_trial_scores(results: Sequence[TrialResult]) -> list[float]:
    scores: list[float] = []
    for r in results:
        if not r.ok or not r.graders:
            continue
        scores.append(sum(g.score for g in r.graders) / len(r.graders))
    return scores


def _compute_judge_stats(results: Sequence[TrialResult], *, max_examples: int = 5) -> dict[str, Any]:
    llm = {"count": 0, "pass": 0, "fail": 0, "unknown": 0}
    pairwise = {"count": 0, "candidate": 0, "baseline": 0, "tie": 0, "unknown": 0, "tie_examples": []}
    multi = {
        "count": 0,
        "agreement_rate_sum": 0.0,
        "agreement_rate_count": 0,
        "disagreements": 0,
        "disagreement_examples": [],
    }

    for r in results:
        for g in r.graders:
            name = getattr(g, "name", "")
            details = g.details or {}
            if name == "llm_judge":
                llm["count"] += 1
                verdict = str(details.get("verdict", "")).lower()
                if verdict == "pass":
                    llm["pass"] += 1
                elif verdict == "fail":
                    llm["fail"] += 1
                else:
                    llm["unknown"] += 1
            elif name == "pairwise_judge":
                pairwise["count"] += 1
                verdict = str(details.get("verdict", "")).lower()
                if verdict == "candidate":
                    pairwise["candidate"] += 1
                elif verdict == "baseline":
                    pairwise["baseline"] += 1
                elif verdict == "tie":
                    pairwise["tie"] += 1
                    if len(pairwise["tie_examples"]) < max_examples:
                        pairwise["tie_examples"].append(r.trial_id)
                else:
                    pairwise["unknown"] += 1
            elif name == "multi_llm_judge":
                multi["count"] += 1
                agreement_rate = details.get("agreement_rate")
                if isinstance(agreement_rate, (int, float)):
                    multi["agreement_rate_sum"] += float(agreement_rate)
                    multi["agreement_rate_count"] += 1
                judges = details.get("judges")
                if isinstance(judges, list) and judges:
                    passes = [bool(j.get("passed")) for j in judges if isinstance(j, dict)]
                    if passes:
                        agreed = all(passes) or all(not p for p in passes)
                        if not agreed:
                            multi["disagreements"] += 1
                            if len(multi["disagreement_examples"]) < max_examples:
                                multi["disagreement_examples"].append(r.trial_id)

    out: dict[str, Any] = {}
    if llm["count"] > 0:
        out["llm_judge"] = {
            "count": llm["count"],
            "pass_rate": llm["pass"] / llm["count"],
            "unknown_rate": llm["unknown"] / llm["count"],
        }
    if pairwise["count"] > 0:
        out["pairwise_judge"] = {
            "count": pairwise["count"],
            "candidate_rate": pairwise["candidate"] / pairwise["count"],
            "baseline_rate": pairwise["baseline"] / pairwise["count"],
            "tie_rate": pairwise["tie"] / pairwise["count"],
            "unknown_rate": pairwise["unknown"] / pairwise["count"],
            "tie_examples": pairwise["tie_examples"],
        }
    if multi["count"] > 0:
        agreement_rate = None
        if multi["agreement_rate_count"] > 0:
            agreement_rate = multi["agreement_rate_sum"] / float(multi["agreement_rate_count"])
        out["multi_llm_judge"] = {
            "count": multi["count"],
            "agreement_rate_mean": agreement_rate,
            "disagreement_rate": multi["disagreements"] / float(multi["count"]),
            "disagreement_examples": multi["disagreement_examples"],
        }
    return out


def _resolve_status(cancel_path: Path, results: Sequence[TrialResult]) -> RunStatus:
    if cancel_path.exists() or any((r.error or "").lower() == "cancelled" for r in results):
        return RunStatus.cancelled
    if any((not r.ok) and (r.error or "").lower() != "cancelled" for r in results):
        return RunStatus.failed
    return RunStatus.completed


def _finalize_run(
    *,
    writer: ArtifactWriter,
    cfg: RunConfig,
    tasks: Sequence[Task],
    results: Sequence[TrialResult],
    cancel_path: Path,
    run_started_at: datetime,
) -> RunSummary:
    passed, failed, avg_score, pass_at_k, pass_pow_k, per_task = _compute_stats(results, tasks)
    summary = RunSummary(
        run_id=cfg.run_id,
        suite_id=cfg.suite.id,
        started_at=run_started_at,
        task_count=len(tasks),
        trial_count=len(results),
        passed_trials=passed,
        failed_trials=failed,
        avg_score=avg_score,
        finished_at=utcnow(),
    )

    status = _resolve_status(cancel_path, results)
    lo, hi = _wilson_ci95(passed, max(1, len(results)))
    scores = _collect_trial_scores(results)
    score_var, score_std = _score_variance(scores)
    judge_stats = _compute_judge_stats(results)
    run_result = RunResult(
        run_id=summary.run_id,
        suite_id=summary.suite_id,
        status=status,
        started_at=summary.started_at,
        finished_at=summary.finished_at,
        task_count=summary.task_count,
        trial_count=summary.trial_count,
        passed_trials=summary.passed_trials,
        failed_trials=summary.failed_trials,
        avg_score=summary.avg_score,
        metrics={
            "pass_at_k": pass_at_k,
            "pass_pow_k": pass_pow_k,
            "trial_pass_rate": (passed / len(results)) if results else 0.0,
            "trial_pass_rate_ci95": [lo, hi],
            "score_variance": score_var,
            "score_stddev": score_std,
            "judge": judge_stats or None,
        },
        per_task={
            tid: {
                "trials": int(v["trials"]),
                "passed": int(v["passed"]),
                "pass_rate": (v["passed"] / v["trials"]) if v["trials"] else 0.0,
            }
            for tid, v in per_task.items()
        },
    )

    writer.write_json("run_result.json", run_result.model_dump())
    return summary


def _adapter_version(adapter: AgentAdapter) -> str:
    version_attr = getattr(adapter, "version", None)
    if callable(version_attr):
        try:
            return str(version_attr())
        except Exception:
            return "unknown"
    if isinstance(version_attr, str):
        return version_attr
    return "unknown"


def _min_provider_limit(bundles: Sequence[TaskBundle]) -> int | None:
    limit: int | None = None
    for bundle in bundles:
        candidate = getattr(bundle.env, "provider_concurrency", None)
        if candidate is None:
            continue
        if candidate < 1:
            raise ValueError(
                f"environment.provider_concurrency must be >= 1 for task '{bundle.task.id}'. "
                f"Got {candidate}."
            )
        limit = candidate if limit is None else min(limit, candidate)
    return limit


@asynccontextmanager
async def _maybe_acquire(sem: asyncio.Semaphore | None):
    if sem is None:
        yield
        return
    async with sem:
        yield


async def run_suite(
    *,
    tasks: Sequence[Task],
    adapter: AgentAdapter,
    graders: Sequence[Grader],
    cfg: RunConfig,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[list[TrialResult], RunSummary]:
    # Ensure jobs_dir is absolute so container backends can mount workspace/logs reliably.
    writer = ArtifactWriter(cfg.jobs_dir.expanduser().resolve(), cfg.run_id)
    writer.ensure_run_dirs()
    cancel_path = writer.run_dir / "cancel.json"

    run_cfg = _ensure_run_config(writer, cfg, {"env_type": cfg.env_type.value})
    seeds = run_cfg.suite.seeds or [random.randint(1, 2**31 - 1)]
    if run_cfg.suite.seeds is None:
        run_cfg = run_cfg.model_copy(update={"suite": run_cfg.suite.model_copy(update={"seeds": seeds})})
        writer.write_json("run_config.json", run_cfg.model_dump())
    trials = _seed_trials(tasks, cfg.suite.trials_per_task, seeds)
    _emit_event(
        on_event,
        "run_start",
        run_id=cfg.run_id,
        task_count=len(tasks),
        trial_count=len(trials),
        env_type=cfg.env_type.value,
    )

    sem = asyncio.Semaphore(cfg.suite.concurrency)
    results: list[TrialResult] = []

    if cfg.mode == "resume":
        results, trials = _resume_trials(
            writer, trials, lambda item, attempt: _trial_id(item.id, attempt)
        )

    async def _run_one(task: Task, attempt: int, seed: int) -> TrialResult:
        async with sem:
            if cancel_path.exists():
                _emit_event(on_event, "trial_cancelled", trial_id=_trial_id(task.id, attempt), task_id=task.id)
                return TrialResult(
                    ok=False,
                    task_id=task.id,
                    trial_id=_trial_id(task.id, attempt),
                    adapter=cfg.adapter_name,
                    seed=seed,
                    started_at=utcnow(),
                    graders=[],
                    error="cancelled",
                )

            trial_id = _trial_id(task.id, attempt)
            _emit_event(on_event, "trial_start", trial_id=trial_id, task_id=task.id)
            writer.ensure_trial_dirs(trial_id)
            trial_cfg = TrialConfig(
                run_id=cfg.run_id,
                trial_id=trial_id,
                task_id=task.id,
                seed=seed,
                attempt=attempt,
                adapter=cfg.adapter_name,
                provenance={"env_type": cfg.env_type.value},
            )
            writer.write_trial_config(trial_cfg)

            started_at = trial_cfg.started_at
            transcript = []
            outcome: Outcome = Outcome(summary=None, data={})
            err: str | None = None
            ok = False

            trial_dir = writer.trial_dir(trial_id)
            workspace_dir = trial_dir / "workspace"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            replay_log = trial_dir / "replay" / "tool_calls.jsonl"
            error_path = trial_dir / "error.txt"

            base_env = LocalToolExecutor(root=workspace_dir)
            env = base_env
            recorder: RecordingToolExecutor | None = None
            if cfg.replay_mode == "record":
                recorder = RecordingToolExecutor(base_env, log_path=replay_log)
                env = recorder
            elif cfg.replay_mode == "replay":
                env = ReplayingToolExecutor(log_path=replay_log, strict=cfg.strict_replay)

            artifacts = TrialArtifacts(writer.trial_dir(trial_id))
            try:
                _emit_event(on_event, "adapter_start", trial_id=trial_id, task_id=task.id)
                timeout_s = task.timeout_seconds
                if timeout_s is not None and timeout_s > 0:
                    transcript, outcome = await asyncio.wait_for(
                        _call_adapter(
                            adapter=adapter,
                            task=task,
                            instruction=_best_effort_instruction(task),
                            seed=seed,
                            env=env,
                            artifacts=artifacts,
                        ),
                        timeout=timeout_s,
                    )
                else:
                    transcript, outcome = await _call_adapter(
                        adapter=adapter,
                        task=task,
                        instruction=_best_effort_instruction(task),
                        seed=seed,
                        env=env,
                        artifacts=artifacts,
                    )
                ok = True
                _emit_event(on_event, "adapter_done", trial_id=trial_id, task_id=task.id)
            except asyncio.TimeoutError:
                err = f"timeout after {task.timeout_seconds}s"
                ok = False
                error_path.write_text(err, encoding="utf-8")
                _emit_event(on_event, "trial_error", trial_id=trial_id, task_id=task.id, error=err)
            except Exception as e:  # pragma: no cover
                err = str(e)
                ok = False
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
                _emit_event(on_event, "trial_error", trial_id=trial_id, task_id=task.id, error=err)
            finally:
                if recorder is not None:
                    recorder.close()

            trajectory = transcript_to_trajectory(
                transcript,
                session_id=trial_id,
                agent_name=getattr(adapter, "name", cfg.adapter_name),
                agent_version=_adapter_version(adapter),
                model_name=getattr(adapter, "model_name", None),
            )
            transcript_path = writer.write_transcript(trial_id, transcript)
            trajectory_path = writer.write_trajectory(trial_id, trajectory)
            outcome_path = writer.write_outcome(trial_id, outcome)

            grader_results = []
            if ok:
                _emit_event(on_event, "grading_start", trial_id=trial_id, task_id=task.id)
                for g in graders:
                    try:
                        if supports_env_grading(g):
                            result = await g.grade_with_env(  # type: ignore[attr-defined]
                                task=task,
                                transcript=transcript,
                                outcome=outcome.data,
                                artifacts=artifacts,
                                env=env,
                            )
                        else:
                            result = await g.grade(
                                task=task, transcript=transcript, outcome=outcome.data, artifacts=artifacts
                            )
                    except Exception as exc:  # pragma: no cover
                        artifacts.verifier().write_json(
                            f"grader_error_{getattr(g, 'name', 'unknown')}.json",
                            {"error": str(exc)},
                        )
                        result = GraderResult(
                            name=getattr(g, "name", "unknown"),
                            score=0.0,
                            passed=False,
                            severity=Severity.error,
                            details={"error": str(exc)},
                        )
                    grader_results.append(result)
                _emit_event(on_event, "grading_done", trial_id=trial_id, task_id=task.id)

            trial_result = TrialResult(
                ok=ok,
                task_id=task.id,
                trial_id=trial_id,
                adapter=cfg.adapter_name,
                seed=seed,
                started_at=started_at,
                transcript_path=str(transcript_path.relative_to(writer.run_dir)),
                trajectory_path=str(trajectory_path.relative_to(writer.run_dir)),
                outcome_path=str(outcome_path.relative_to(writer.run_dir)),
                graders=grader_results,
                error=err,
            )
            writer.write_result(trial_id, trial_result)
            _emit_event(
                on_event,
                "trial_done",
                trial_id=trial_id,
                task_id=task.id,
                ok=ok,
                error=err,
            )
            return trial_result

    if cancel_path.exists():
        new_results: list[TrialResult] = []
    else:
        new_results = list(await asyncio.gather(*[_run_one(t, a, s) for (t, a, s) in trials]))
    results = results + new_results

    summary = _finalize_run(
        writer=writer,
        cfg=cfg,
        tasks=tasks,
        results=results,
        cancel_path=cancel_path,
        run_started_at=run_cfg.started_at,
    )
    _emit_event(on_event, "run_done", run_id=cfg.run_id)
    return results, summary


async def run_suite_bundles(
    *,
    bundles: Sequence[TaskBundle],
    adapter: AgentAdapter,
    graders: Sequence[Grader],
    cfg: RunConfig,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[list[TrialResult], RunSummary]:
    writer = ArtifactWriter(cfg.jobs_dir.expanduser().resolve(), cfg.run_id)
    writer.ensure_run_dirs()
    cancel_path = writer.run_dir / "cancel.json"
    run_cfg = _ensure_run_config(writer, cfg, {"env_type": cfg.env_type.value})
    seeds = run_cfg.suite.seeds or [random.randint(1, 2**31 - 1)]
    if run_cfg.suite.seeds is None:
        run_cfg = run_cfg.model_copy(update={"suite": run_cfg.suite.model_copy(update={"seeds": seeds})})
        writer.write_json("run_config.json", run_cfg.model_dump())
    compatibility_issues = collect_compatibility_issues(bundles, cfg.env_type)
    if compatibility_issues:
        sample = compatibility_issues[:5]
        lines = "\n".join([f"- {i.task_id}: {i.reason}" for i in sample])
        extra = ""
        if len(compatibility_issues) > len(sample):
            extra = f"\n... and {len(compatibility_issues) - len(sample)} more task(s)."
        raise RuntimeError(
            "Selected environment is incompatible with one or more task bundles:\n"
            f"{lines}{extra}\n"
            "Tip: use --env docker_cli/daytona for Dockerfile-based datasets."
        )
    trials = _seed_trials(bundles, cfg.suite.trials_per_task, seeds)
    _emit_event(
        on_event,
        "run_start",
        run_id=cfg.run_id,
        task_count=len(bundles),
        trial_count=len(trials),
        env_type=cfg.env_type.value,
    )

    sem = asyncio.Semaphore(cfg.suite.concurrency)
    provider_limit = (
        _min_provider_limit(bundles)
        if cfg.env_type in (EnvironmentType.daytona, EnvironmentType.e2b)
        else None
    )
    provider_sem = asyncio.Semaphore(provider_limit) if provider_limit else None
    results: list[TrialResult] = []

    if cfg.mode == "resume":
        results, trials = _resume_trials(
            writer, trials, lambda item, attempt: _trial_id(item.task.id, attempt)
        )

    async def _run_one(bundle: TaskBundle, attempt: int, seed: int) -> TrialResult:
        async with _maybe_acquire(provider_sem):
            async with sem:
                if cancel_path.exists():
                    _emit_event(
                        on_event, "trial_cancelled", trial_id=_trial_id(bundle.task.id, attempt), task_id=bundle.task.id
                    )
                    return TrialResult(
                        ok=False,
                        task_id=bundle.task.id,
                        trial_id=_trial_id(bundle.task.id, attempt),
                        adapter=cfg.adapter_name,
                        seed=seed,
                        started_at=utcnow(),
                        graders=[],
                        error="cancelled",
                    )

                trial_id = _trial_id(bundle.task.id, attempt)
                _emit_event(on_event, "trial_start", trial_id=trial_id, task_id=bundle.task.id)
                writer.ensure_trial_dirs(trial_id)
                transcript = []
                outcome: Outcome = Outcome(summary=None, data={})
                err: str | None = None
                ok = False

                trial_dir = writer.trial_dir(trial_id)
                workspace_dir = trial_dir / "workspace"
                logs_dir = trial_dir / "env_logs"
                replay_log = trial_dir / "replay" / "tool_calls.jsonl"
                error_path = trial_dir / "error.txt"
                logs_dir.mkdir(parents=True, exist_ok=True)

                env_cfg = EnvironmentConfig(
                    allow_internet=bundle.env.allow_internet,
                    cpus=bundle.env.resources.cpus,
                    memory_mb=bundle.env.resources.memory_mb,
                    storage_mb=bundle.env.resources.storage_mb,
                    gpus=bundle.env.resources.gpus,
                    build_timeout_sec=bundle.env.build_timeout_sec,
                    provider_concurrency=bundle.env.provider_concurrency,
                )


                env_type = cfg.env_type
                container_image = None
                if getattr(bundle.env, "type", None) == "container":
                    container_image = getattr(bundle.env, "image", None)
                container_platform = None
                if getattr(bundle.env, "type", None) == "container":
                    container_platform = getattr(bundle.env, "platform", None)
                container_workdir = None
                container_workspace_mount = None
                if getattr(bundle.env, "type", None) == "container":
                    container_workdir = getattr(bundle.env, "workdir", None)
                    container_workspace_mount = getattr(bundle.env, "workspace_mount", None)

                trial_cfg = TrialConfig(
                    run_id=cfg.run_id,
                    trial_id=trial_id,
                    task_id=bundle.task.id,
                    seed=seed,
                    attempt=attempt,
                    adapter=cfg.adapter_name,
                    provenance={
                        "env_type": env_type.value,
                        "task_bundle_digest": bundle.digest,
                        "container_image": container_image,
                        "allow_internet": bundle.env.allow_internet,
                        "resources": bundle.env.resources.model_dump(),
                    },
                )
                writer.write_trial_config(trial_cfg)

                started_at = trial_cfg.started_at

                environment = EnvironmentFactory.create(
                    env_type=env_type,
                    trial_id=trial_id,
                    workspace_dir=workspace_dir,
                    logs_dir=logs_dir,
                    cfg=env_cfg,
                    task_environment_dir=bundle.environment_dir,
                    task_files_dir=bundle.files_dir,
                    task_digest=bundle.digest,
                    container_image=container_image,
                    container_platform=container_platform,
                    container_workdir=container_workdir,
                    container_workspace_mount=container_workspace_mount,
                    docker_image_tag_prefix=cfg.docker_image_tag_prefix,
                    daytona_snapshot_template=cfg.daytona_snapshot_template,
                    daytona_network_block_all=cfg.daytona_network_block_all,
                )

                recorder: RecordingToolExecutor | None = None
                tool_env = EnvironmentToolExecutor(environment)
                tool_executor = tool_env
                if cfg.replay_mode == "record":
                    recorder = RecordingToolExecutor(tool_env, log_path=replay_log)
                    tool_executor = recorder
                elif cfg.replay_mode == "replay":
                    tool_executor = ReplayingToolExecutor(log_path=replay_log, strict=cfg.strict_replay)

                grader_results = []
                try:
                    _emit_event(on_event, "env_start", trial_id=trial_id, task_id=bundle.task.id)
                    await environment.start(force_build=False)
                    _emit_event(on_event, "env_ready", trial_id=trial_id, task_id=bundle.task.id)
               
                    if bundle.files_dir is not None and bundle.files_dir.exists():
                        await environment.upload_dir(bundle.files_dir, ".")
                    if bundle.tests_dir is not None and bundle.tests_dir.exists():
                        await environment.upload_dir(bundle.tests_dir, "/tests")
                    
                        await environment.exec("sh -lc 'chmod +x /tests/test.sh 2>/dev/null || true'")

                    artifacts = TrialArtifacts(writer.trial_dir(trial_id))
                    _emit_event(on_event, "adapter_start", trial_id=trial_id, task_id=bundle.task.id)
                    timeout_s = bundle.task.timeout_seconds
                    if timeout_s is not None and timeout_s > 0:
                        transcript, outcome = await asyncio.wait_for(
                            _call_adapter(
                                adapter=adapter,
                                task=bundle.task,
                                instruction=bundle.instruction or _best_effort_instruction(bundle.task),
                                seed=seed,
                                env=tool_executor,
                                artifacts=artifacts,
                            ),
                            timeout=timeout_s,
                        )
                    else:
                        transcript, outcome = await _call_adapter(
                            adapter=adapter,
                            task=bundle.task,
                            instruction=bundle.instruction or _best_effort_instruction(bundle.task),
                            seed=seed,
                            env=tool_executor,
                            artifacts=artifacts,
                        )
                    ok = True
                    _emit_event(on_event, "adapter_done", trial_id=trial_id, task_id=bundle.task.id)
                    if ok:
                        _emit_event(on_event, "grading_start", trial_id=trial_id, task_id=bundle.task.id)
                        for g in graders:
                            try:
                                if supports_env_grading(g):
                                    result = await g.grade_with_env(  # type: ignore[attr-defined]
                                        task=bundle.task,
                                        transcript=transcript,
                                        outcome=outcome.data,
                                        artifacts=artifacts,
                                        env=tool_executor,
                                    )
                                else:
                                    result = await g.grade(
                                        task=bundle.task,
                                        transcript=transcript,
                                        outcome=outcome.data,
                                        artifacts=artifacts,
                                    )
                            except Exception as exc:  # pragma: no cover
                                artifacts.verifier().write_json(
                                    f"grader_error_{getattr(g, 'name', 'unknown')}.json",
                                    {"error": str(exc)},
                                )
                                result = GraderResult(
                                    name=getattr(g, "name", "unknown"),
                                    score=0.0,
                                    passed=False,
                                    severity=Severity.error,
                                    details={"error": str(exc)},
                            )
                            grader_results.append(result)
                        _emit_event(on_event, "grading_done", trial_id=trial_id, task_id=bundle.task.id)
                except asyncio.TimeoutError:
                    err = f"timeout after {bundle.task.timeout_seconds}s"
                    ok = False
                    error_path.write_text(err, encoding="utf-8")
                    _emit_event(on_event, "trial_error", trial_id=trial_id, task_id=bundle.task.id, error=err)
                except Exception as e:  # pragma: no cover
                    err = str(e)
                    ok = False
                    error_path.write_text(traceback.format_exc(), encoding="utf-8")
                    _emit_event(on_event, "trial_error", trial_id=trial_id, task_id=bundle.task.id, error=err)
                finally:
                    try:
                        await environment.stop(delete=True)
                    except Exception:  # pragma: no cover
                        cleanup_error = logs_dir / "cleanup_error.txt"
                        cleanup_error.write_text(traceback.format_exc(), encoding="utf-8")
                    if recorder is not None:
                        recorder.close()

                trajectory = transcript_to_trajectory(
                    transcript,
                    session_id=trial_id,
                    agent_name=getattr(adapter, "name", cfg.adapter_name),
                    agent_version=_adapter_version(adapter),
                    model_name=getattr(adapter, "model_name", None),
                )
                transcript_path = writer.write_transcript(trial_id, transcript)
                trajectory_path = writer.write_trajectory(trial_id, trajectory)
                outcome_path = writer.write_outcome(trial_id, outcome)

                trial_result = TrialResult(
                    ok=ok,
                    task_id=bundle.task.id,
                    trial_id=trial_id,
                    adapter=cfg.adapter_name,
                    seed=seed,
                    started_at=started_at,
                    transcript_path=str(transcript_path.relative_to(writer.run_dir)),
                    trajectory_path=str(trajectory_path.relative_to(writer.run_dir)),
                    outcome_path=str(outcome_path.relative_to(writer.run_dir)),
                    graders=grader_results,
                    error=err,
                )
                writer.write_result(trial_id, trial_result)
                _emit_event(
                    on_event,
                    "trial_done",
                    trial_id=trial_id,
                    task_id=bundle.task.id,
                    ok=ok,
                    error=err,
                )
                return trial_result

    if cancel_path.exists():
        new_results: list[TrialResult] = []
    else:
        new_results = list(await asyncio.gather(*[_run_one(b, a, s) for (b, a, s) in trials]))
    results = results + new_results

    tasks = [b.task for b in bundles]
    summary = _finalize_run(
        writer=writer,
        cfg=cfg,
        tasks=tasks,
        results=results,
        cancel_path=cancel_path,
        run_started_at=run_cfg.started_at,
    )
    _emit_event(on_event, "run_done", run_id=cfg.run_id)
    return results, summary
