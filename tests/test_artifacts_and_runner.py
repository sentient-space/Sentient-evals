import asyncio
import json
from pathlib import Path

from sentient_evals.artifacts import ArtifactWriter, TrialArtifacts
from sentient_evals.env import ExecResult, ToolExecutor
from sentient_evals.graders import ExactMatchGrader
from sentient_evals.atif.converters import transcript_to_trajectory
from sentient_evals.models import GraderResult, Outcome, SuiteConfig, Task, TranscriptEvent
from sentient_evals.runner import RunConfig, run_suite, run_suite_bundles
from sentient_evals.replay import RecordingToolExecutor, ReplayingToolExecutor
from sentient_evals.task_bundles import load_task_bundles


class FakeAdapter:
    name = "fake"

    async def run(self, task: Task, *, seed: int, env):
        return (
            [TranscriptEvent(kind="message", role="user", content="hi")],
            Outcome(summary="ok", data={"answer": "x"}),
        )


def test_artifact_writer_roundtrip(tmp_path: Path):
    w = ArtifactWriter(tmp_path, "run1")
    w.ensure_run_dirs()
    p = w.write_json("meta.json", {"ok": True})
    assert p.exists()
    assert json.loads(p.read_text())["ok"] is True


def test_run_suite_writes_result(tmp_path: Path):
    tasks = [Task(id="t1", input={"q": "?"})]
    suite = SuiteConfig(id="s", trials_per_task=1, concurrency=1, seeds=[123])
    cfg = RunConfig(run_id="r1", suite=suite, jobs_dir=tmp_path, adapter_name="fake")
    graders = [ExactMatchGrader(field="answer", expected="x")]

    results, summary = asyncio.run(
        run_suite(tasks=tasks, adapter=FakeAdapter(), graders=graders, cfg=cfg)
    )

    assert len(results) == 1
    assert summary.passed_trials == 1
    assert (tmp_path / "r1" / "config.json").exists()
    assert (tmp_path / "r1" / "result.json").exists()
    assert (tmp_path / "r1" / "trials" / "t1__0" / "judge").is_dir()
    assert (tmp_path / "r1" / "trials" / "t1__0" / "verifier").is_dir()
    assert (tmp_path / "r1" / "trials" / "t1__0" / "trajectory.json").exists()


class ArtifactWritingGrader:
    name = "artifact_writing"

    async def grade(self, *, task: Task, transcript, outcome, artifacts: TrialArtifacts):
        artifacts.judge().write_json("hello.json", {"ok": True})
        return GraderResult(name=self.name, score=1.0, passed=True)


def test_grader_can_write_judge_artifacts(tmp_path: Path):
    tasks = [Task(id="t1", input={"q": "?"})]
    suite = SuiteConfig(id="s", trials_per_task=1, concurrency=1, seeds=[123])
    cfg = RunConfig(run_id="r1", suite=suite, jobs_dir=tmp_path, adapter_name="fake")

    results, _summary = asyncio.run(
        run_suite(tasks=tasks, adapter=FakeAdapter(), graders=[ArtifactWritingGrader()], cfg=cfg)
    )
    assert results[0].ok is True
    assert (tmp_path / "r1" / "trials" / "t1__0" / "judge" / "hello.json").exists()


def test_junit_export(tmp_path: Path):
    from sentient_evals.junit import JUnitExportConfig, trials_to_junit_xml
    from sentient_evals.models import TrialResult

    tr = TrialResult(
        ok=True,
        task_id="t1",
        trial_id="t1__0",
        adapter="fake",
        seed=1,
        started_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        graders=[],
    )
    xml = trials_to_junit_xml([tr], JUnitExportConfig(suite_name="suite"))
    assert "<testsuite" in xml
    assert "suite" in xml


class FakeToolExecutor:
    def __init__(self):
        self.files: dict[str, str] = {}

    async def exec(self, cmd: str, *, timeout_s: float | None = None) -> ExecResult:
        return ExecResult(stdout=f"ran:{cmd}", stderr="", exit_code=0, duration_ms=1)

    async def read_file(self, path: str) -> str:
        return self.files.get(path, "")

    async def write_file(self, path: str, content: str) -> None:
        self.files[path] = content

    async def list_dir(self, path: str) -> list[str]:
        return []

    async def call(self, tool_name: str, args: dict) -> dict:
        return {"tool": tool_name, "args": args}


def test_tool_record_replay_roundtrip(tmp_path: Path):
    log = tmp_path / "tool_calls.jsonl"
    base: ToolExecutor = FakeToolExecutor()
    rec = RecordingToolExecutor(base, log_path=log)
    out = asyncio.run(rec.exec("echo hi"))
    assert out.stdout == "ran:echo hi"
    rec.close()

    rep = ReplayingToolExecutor(log_path=log, strict=True)
    out2 = asyncio.run(rep.exec("echo hi"))
    assert out2.stdout == "ran:echo hi"


def test_metrics_custom_fields_are_preserved_in_atif():
    transcript = [
        TranscriptEvent(
            kind="metric",
            metrics={"workflow_steps": 1.0, "prompt_tokens": 3},
            content="route=default",
        )
    ]
    trajectory = transcript_to_trajectory(
        transcript,
        session_id="s1",
        agent_name="test",
        agent_version="0.0.0",
    )
    metrics = trajectory.steps[0].metrics
    assert metrics is not None
    assert metrics.prompt_tokens == 3
    assert metrics.extra == {"workflow_steps": 1.0}


def test_resume_skips_completed_trials(tmp_path: Path):
    tasks = [Task(id="t1", input={"q": "?"})]
    suite = SuiteConfig(id="s", trials_per_task=2, concurrency=1, seeds=[123, 456])
    run_id = "r1"

    # Pre-create trial 0 result to simulate partial completion.
    w = ArtifactWriter(tmp_path, run_id)
    w.ensure_run_dirs()
    w.ensure_trial_dirs("t1__0")
    from sentient_evals.models import TrialResult

    w.write_result(
        "t1__0",
        TrialResult(
            ok=True,
            task_id="t1",
            trial_id="t1__0",
            adapter="fake",
            seed=123,
            started_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            graders=[],
        ),
    )

    cfg = RunConfig(run_id=run_id, suite=suite, jobs_dir=tmp_path, adapter_name="fake", mode="resume")
    results, _summary = asyncio.run(run_suite(tasks=tasks, adapter=FakeAdapter(), graders=[], cfg=cfg))
    # One existing + one newly executed
    assert len(results) == 2
    assert (tmp_path / run_id / "trials" / "t1__1" / "result.json").exists()


def test_cancel_prevents_scheduling(tmp_path: Path):
    tasks = [Task(id="t1", input={"q": "?"})]
    suite = SuiteConfig(id="s", trials_per_task=3, concurrency=1, seeds=[1, 2, 3])
    run_id = "r1"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cancel.json").write_text("{}", encoding="utf-8")

    cfg = RunConfig(run_id=run_id, suite=suite, jobs_dir=tmp_path, adapter_name="fake", mode="fresh")
    results, _summary = asyncio.run(run_suite(tasks=tasks, adapter=FakeAdapter(), graders=[], cfg=cfg))
    assert results == []


class ReadsWorkspaceAdapter:
    name = "reads_workspace"

    async def run(self, task: Task, *, seed: int, env):
        txt = await env.read_file("hello.txt")
        return (
            [TranscriptEvent(kind="message", role="user", content=task.id)],
            Outcome(summary="ok", data={"answer": txt.strip()}),
        )


def test_run_suite_bundles_local_python(tmp_path: Path):
    task_dir = tmp_path / "tasks" / "t1"
    (task_dir / "files").mkdir(parents=True)
    (task_dir / "files" / "hello.txt").write_text("world\n", encoding="utf-8")
    (task_dir / "task.toml").write_text(
        '\n'.join(
            [
                'id = "t1"',
                "",
                "[environment]",
                'type = "local_python"',
                "",
                "[input]",
                'q = "?"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (task_dir / "instruction.md").write_text("do thing", encoding="utf-8")

    bundles = load_task_bundles(task_dir.parent)
    assert len(bundles) == 1
    assert bundles[0].digest

    suite = SuiteConfig(id="s", trials_per_task=1, concurrency=1, seeds=[123])
    cfg = RunConfig(run_id="r1", suite=suite, jobs_dir=tmp_path, adapter_name="reads_workspace")
    graders = [ExactMatchGrader(field="answer", expected="world")]

    results, _summary = asyncio.run(
        run_suite_bundles(bundles=bundles, adapter=ReadsWorkspaceAdapter(), graders=graders, cfg=cfg)
    )
    assert len(results) == 1

    trial_cfg = json.loads((tmp_path / "r1" / "trials" / "t1__0" / "config.json").read_text(encoding="utf-8"))
    assert trial_cfg["provenance"]["env_type"] == "local_python"
    assert trial_cfg["provenance"]["task_bundle_digest"] == bundles[0].digest

