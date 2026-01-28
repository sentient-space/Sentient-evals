from __future__ import annotations

import asyncio
import random
import traceback
from dataclasses import dataclass
from pathlib import Path
from math import sqrt
from typing import Callable, Literal, Sequence, TypeVar

from .adapters import AgentAdapter
from .atif.converters import transcript_to_trajectory
from . import __version__
from .artifacts import ArtifactWriter, TrialArtifacts
from .env import EnvironmentToolExecutor, LocalToolExecutor
from .environments.base import EnvironmentConfig, EnvironmentType
from .environments.factory import EnvironmentFactory
from .graders import Grader, supports_env_grading
from .models import (
    Outcome,
    Severity,
    RunConfigFile,
    RunResult,
    RunStatus,
    RunSummary,
    SuiteConfig,
    Task,
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


def _ensure_run_config(
    writer: ArtifactWriter, cfg: RunConfig, extra_provenance: dict[str, object] | None
) -> None:
    if cfg.mode == "fresh" or not (writer.run_dir / "config.json").exists():
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
        writer.write_json("config.json", run_cfg.model_dump())


T = TypeVar("T")


def _seed_trials(items: Sequence[T], trials_per_task: int, seeds: list[int]) -> list[tuple[T, int, int]]:
    trials: list[tuple[T, int, int]] = []
    for item in items:
        for attempt in range(trials_per_task):
            seed = seeds[attempt % len(seeds)]
            trials.append((item, attempt, seed))
    return trials


def _resume_trials(
    writer: ArtifactWriter,
    trials: list[tuple[T, int, int]],
    trial_id_fn: Callable[[T, int], str],
) -> tuple[list[TrialResult], list[tuple[T, int, int]]]:
    results: list[TrialResult] = []
    remaining: list[tuple[T, int, int]] = []
    for item, attempt, seed in trials:
        tid = trial_id_fn(item, attempt)
        existing = writer.trial_dir(tid) / "result.json"
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
) -> RunSummary:
    passed, failed, avg_score, pass_at_k, pass_pow_k, per_task = _compute_stats(results, tasks)
    summary = RunSummary(
        run_id=cfg.run_id,
        suite_id=cfg.suite.id,
        task_count=len(tasks),
        trial_count=len(results),
        passed_trials=passed,
        failed_trials=failed,
        avg_score=avg_score,
        finished_at=utcnow(),
    )

    status = _resolve_status(cancel_path, results)
    lo, hi = _wilson_ci95(passed, max(1, len(results)))
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

    writer.write_json("result.json", run_result.model_dump())
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

async def run_suite(
    *,
    tasks: Sequence[Task],
    adapter: AgentAdapter,
    graders: Sequence[Grader],
    cfg: RunConfig,
) -> tuple[list[TrialResult], RunSummary]:
    writer = ArtifactWriter(cfg.jobs_dir, cfg.run_id)
    writer.ensure_run_dirs()
    cancel_path = writer.run_dir / "cancel.json"
    _ensure_run_config(writer, cfg, None)
    seeds = cfg.suite.seeds or [random.randint(1, 2**31 - 1)]
    trials = _seed_trials(tasks, cfg.suite.trials_per_task, seeds)

    sem = asyncio.Semaphore(cfg.suite.concurrency)
    results: list[TrialResult] = []

    if cfg.mode == "resume":
        results, trials = _resume_trials(
            writer, trials, lambda item, attempt: _trial_id(item.id, attempt)
        )

    async def _run_one(task: Task, attempt: int, seed: int) -> TrialResult:
        async with sem:
            if cancel_path.exists():
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

            try:
                transcript, outcome = await adapter.run(task, seed=seed, env=env)
                ok = True
            except Exception as e:  # pragma: no cover
                err = str(e)
                ok = False
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
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
            trajectory_path = writer.write_trajectory(trial_id, trajectory)
            outcome_path = writer.write_outcome(trial_id, outcome)

            grader_results = []
            if ok:
                artifacts = TrialArtifacts(writer.trial_dir(trial_id))
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

            trial_result = TrialResult(
                ok=ok,
                task_id=task.id,
                trial_id=trial_id,
                adapter=cfg.adapter_name,
                seed=seed,
                started_at=started_at,
                trajectory_path=str(trajectory_path.relative_to(writer.run_dir)),
                outcome_path=str(outcome_path.relative_to(writer.run_dir)),
                graders=grader_results,
                error=err,
            )
            writer.write_result(trial_id, trial_result)
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
    )
    return results, summary


async def run_suite_bundles(
    *,
    bundles: Sequence[TaskBundle],
    adapter: AgentAdapter,
    graders: Sequence[Grader],
    cfg: RunConfig,
) -> tuple[list[TrialResult], RunSummary]:
    writer = ArtifactWriter(cfg.jobs_dir, cfg.run_id)
    writer.ensure_run_dirs()
    cancel_path = writer.run_dir / "cancel.json"
    _ensure_run_config(writer, cfg, {"env_type": cfg.env_type.value})
    seeds = cfg.suite.seeds or [random.randint(1, 2**31 - 1)]
    trials = _seed_trials(bundles, cfg.suite.trials_per_task, seeds)

    sem = asyncio.Semaphore(cfg.suite.concurrency)
    results: list[TrialResult] = []

    if cfg.mode == "resume":
        results, trials = _resume_trials(
            writer, trials, lambda item, attempt: _trial_id(item.task.id, attempt)
        )

    async def _run_one(bundle: TaskBundle, attempt: int, seed: int) -> TrialResult:
        async with sem:
            if cancel_path.exists():
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
            )

            env_type = cfg.env_type
            if bundle.env.type == "local_python":
                env_type = EnvironmentType.local_python
            container_image = None
            if getattr(bundle.env, "type", None) == "container":
                container_image = getattr(bundle.env, "image", None)

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
                await environment.start(force_build=False)
                transcript, outcome = await adapter.run(bundle.task, seed=seed, env=tool_executor)
                ok = True
                if ok:
                    artifacts = TrialArtifacts(writer.trial_dir(trial_id))
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
            except Exception as e:  # pragma: no cover
                err = str(e)
                ok = False
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
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
            trajectory_path = writer.write_trajectory(trial_id, trajectory)
            outcome_path = writer.write_outcome(trial_id, outcome)

            trial_result = TrialResult(
                ok=ok,
                task_id=bundle.task.id,
                trial_id=trial_id,
                adapter=cfg.adapter_name,
                seed=seed,
                started_at=started_at,
                trajectory_path=str(trajectory_path.relative_to(writer.run_dir)),
                outcome_path=str(outcome_path.relative_to(writer.run_dir)),
                graders=grader_results,
                error=err,
            )
            writer.write_result(trial_id, trial_result)
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
    )
    return results, summary

