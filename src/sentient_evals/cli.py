from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .environments.base import EnvironmentType
from .config_files import load_run_spec
from .models import SuiteConfig, Task
from .junit import JUnitExportConfig, trials_to_junit_xml
from .schema_export import export_json_schemas
from .runner import RunConfig, run_suite, run_suite_bundles
from .task_bundles import load_task_bundles
from .registry import build_adapter, build_grader

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command()
def run(
    tasks_path: Optional[Path] = typer.Option(None, "--tasks", exists=True, readable=True),
    tasks_dir: Optional[Path] = typer.Option(None, "--tasks-dir", exists=True, readable=True),
    config: Optional[Path] = typer.Option(None, "--config", exists=True, readable=True),
    graders_file: Optional[Path] = typer.Option(None, "--graders-file", exists=True, readable=True),
    adapter: Optional[str] = typer.Option(None, "--adapter"),
    adapter_kwargs: Optional[str] = typer.Option(None, "--adapter-kwargs"),
    jobs_dir: Path = typer.Option(Path("jobs"), "--jobs-dir"),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
    suite_id: str = typer.Option("default", "--suite-id"),
    trials_per_task: int = typer.Option(1, "--trials-per-task", min=1),
    concurrency: int = typer.Option(1, "--concurrency", min=1),
    seed: Optional[int] = typer.Option(None, "--seed"),
    resume: bool = typer.Option(False, "--resume"),
    replay_mode: str = typer.Option("off", "--replay-mode", case_sensitive=False),
    env: str = typer.Option("local_python", "--env", case_sensitive=False),
    docker_image_tag_prefix: str = typer.Option("sentient-evals", "--docker-image-tag-prefix"),
    daytona_snapshot_template: Optional[str] = typer.Option(None, "--daytona-snapshot-template"),
    daytona_network_block_all: Optional[bool] = typer.Option(None, "--daytona-network-block-all"),
):
    """
    Run a local eval suite from a tasks JSON file or a task bundles directory.

    tasks JSON format:
      [{"id": "...", "input": {...}, "metadata": {...}}]
    """

    if (tasks_path is None) == (tasks_dir is None):
        raise typer.BadParameter("Provide exactly one of --tasks or --tasks-dir")
    if tasks_path is not None and env.lower() != EnvironmentType.local_python.value:
        raise typer.BadParameter("--env only supports local_python when using --tasks")

    spec = load_run_spec(config) if config is not None else None
    suite = spec.suite if spec and spec.suite is not None else SuiteConfig(id=suite_id)
    suite = suite.model_copy(
        update={
            "id": suite_id,
            "trials_per_task": trials_per_task,
            "concurrency": concurrency,
            "seeds": [seed] if seed is not None else suite.seeds,
        }
    )

    if adapter is not None:
        if adapter in {"workflow_stub", "example_installed"}:
            adapter_type = adapter
            import_path = None
        else:
            adapter_type = "import"
            import_path = adapter
        kwargs = json.loads(adapter_kwargs) if adapter_kwargs else {}
        adapter_obj = build_adapter(adapter_type=adapter_type, import_path=import_path, kwargs=kwargs)
    elif spec is not None:
        adapter_obj = build_adapter(
            adapter_type=spec.adapter.type, import_path=spec.adapter.import_path, kwargs=spec.adapter.kwargs
        )
    else:
        raise typer.BadParameter("Provide --config or --adapter to select an adapter")

    grader_specs: list[dict] = []
    if graders_file is not None:
        grader_specs = json.loads(graders_file.read_text(encoding="utf-8"))
    elif spec is not None:
        grader_specs = spec.graders
    elif tasks_dir is not None:
        grader_specs = [{"type": "verifier_script", "config": {}}]
    else:
        raise typer.BadParameter("Provide --graders-file or --config (graders) when using --tasks")

    graders = [build_grader(s) for s in grader_specs]
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S")
    env_type = EnvironmentType(env.lower())
    cfg = RunConfig(
        run_id=run_id,
        suite=suite,
        jobs_dir=jobs_dir,
        adapter_name=getattr(adapter_obj, "name", "adapter"),
        mode="resume" if resume else "fresh",
        replay_mode=replay_mode.lower(),  # type: ignore[arg-type]
        env_type=env_type,
        docker_image_tag_prefix=docker_image_tag_prefix,
        daytona_snapshot_template=daytona_snapshot_template,
        daytona_network_block_all=daytona_network_block_all,
    )

    if tasks_dir is not None:
        bundles = load_task_bundles(tasks_dir)
        results, summary = asyncio.run(
            run_suite_bundles(bundles=bundles, adapter=adapter_obj, graders=graders, cfg=cfg)
        )
    else:
        assert tasks_path is not None
        tasks_raw = json.loads(tasks_path.read_text())
        tasks = [Task(**t) for t in tasks_raw]
        results, summary = asyncio.run(run_suite(tasks=tasks, adapter=adapter_obj, graders=graders, cfg=cfg))

    table = Table(title=f"sentient-evals: {summary.run_id}")
    table.add_column("trial")
    table.add_column("ok")
    table.add_column("passed")
    table.add_column("score")
    for r in results:
        passed = "n/a"
        score = "n/a"
        if r.ok and r.graders:
            passed = str(all(g.passed for g in r.graders))
            score = f"{(sum(g.score for g in r.graders)/len(r.graders)):.2f}"
        table.add_row(r.trial_id, str(r.ok), passed, score)

    console.print(table)
    console.print(f"Wrote artifacts to: {jobs_dir / run_id}")


@app.command()
def cancel(run_dir: Path = typer.Argument(..., exists=True, readable=True)):
    """Request cancellation of an in-progress run by creating cancel.json."""
    p = run_dir / "cancel.json"
    if not p.exists():
        p.write_text(json.dumps({"requested_at": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
    console.print(f"Wrote cancel request: {p}")


@app.command()
def report(run_dir: Path = typer.Argument(..., exists=True, readable=True)):
    """Render a quick summary for an existing run directory."""
    result_path = run_dir / "result.json"
    if not result_path.exists():
        raise typer.BadParameter("result.json not found")
    console.print_json(result_path.read_text())


@app.command()
def junit(
    run_dir: Path = typer.Argument(..., exists=True, readable=True),
    out_file: Path = typer.Option(Path("junit.xml"), "--out"),
    suite_name: str = typer.Option("sentient-evals", "--suite-name"),
):
    """Export a run directory into JUnit XML (for CI)."""
    trials_dir = run_dir / "trials"
    if not trials_dir.exists():
        raise typer.BadParameter("trials/ not found in run directory")

    trial_results = []
    for result_path in sorted(trials_dir.glob("*/result.json")):
        from .models import TrialResult

        trial_results.append(TrialResult.model_validate_json(result_path.read_text()))

    xml = trials_to_junit_xml(trial_results, JUnitExportConfig(suite_name=suite_name))
    out_file.write_text(xml, encoding="utf-8")
    console.print(f"Wrote JUnit XML to {out_file}")


@app.command()
def diff(
    baseline_dir: Path = typer.Argument(..., exists=True, readable=True),
    candidate_dir: Path = typer.Argument(..., exists=True, readable=True),
):
    """
    Compare two runs (baseline vs candidate) using their result.json files.
    """
    b = json.loads((baseline_dir / "result.json").read_text())
    c = json.loads((candidate_dir / "result.json").read_text())

    table = Table(title="sentient-evals diff")
    table.add_column("metric")
    table.add_column("baseline")
    table.add_column("candidate")
    for k in ["passed_trials", "failed_trials", "avg_score", "trial_count", "task_count"]:
        table.add_row(k, str(b.get(k)), str(c.get(k)))
    console.print(table)


schema_app = typer.Typer(no_args_is_help=True)
app.add_typer(schema_app, name="schema")


@schema_app.command("export")
def schema_export(out_dir: Path = typer.Option(Path("schemas/v1"), "--out-dir")):
    """Export versioned JSON Schemas for public contracts."""
    written = export_json_schemas(out_dir)
    console.print(f"Wrote {len(written)} schemas to {out_dir}")

