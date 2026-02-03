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
from .agent_file import load_agent_adapter_from_file, parse_agent_file_ref
from .config_files import load_run_spec
from .models import SuiteConfig, Task
from .junit import JUnitExportConfig, trials_to_junit_xml
from .schema_export import export_json_schemas
from .runner import RunConfig, run_suite, run_suite_bundles
from .task_bundles import load_task_bundles
from .registry import build_adapter, build_grader
from .datasets import DatasetClient, RegistryClientFactory

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

datasets_app = typer.Typer(no_args_is_help=True)
app.add_typer(datasets_app, name="datasets")

tasks_app = typer.Typer(no_args_is_help=True)
app.add_typer(tasks_app, name="tasks")

_BUILTIN_ADAPTERS = {
    "workflow_stub",
    "example_installed",
    "aider",
    "claude-code",
    "codex",
    "cursor-cli",
    "cline-cli",
    "gemini-cli",
    "goose",
    "mini-swe-agent",
    "opencode",
    "openhands",
    "qwen-coder",
    "swe-agent",
}


@app.command()
def run(
    tasks_path: Optional[Path] = typer.Option(None, "--tasks", exists=True, readable=True),
    tasks_dir: Optional[Path] = typer.Option(None, "--tasks-dir", exists=True, readable=True),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Dataset name@version from registry"),
    registry_url: Optional[str] = typer.Option(None, "--registry-url", help="Registry URL for --dataset"),
    registry_path: Optional[Path] = typer.Option(None, "--registry-path", exists=True, help="Registry path for --dataset"),
    config: Optional[Path] = typer.Option(None, "--config", exists=True, readable=True),
    graders_file: Optional[Path] = typer.Option(None, "--graders-file", exists=True, readable=True),
    agent_file: Optional[str] = typer.Option(None, "--agent-file"),
    adapter: Optional[str] = typer.Option(None, "--adapter", "-a", help="Adapter name (gemini-cli, claude-code, etc.) or import path"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name (e.g., google/gemini-2.0-flash)"),
    adapter_kwargs: Optional[str] = typer.Option(None, "--adapter-kwargs", help="JSON dict of additional adapter kwargs"),
    jobs_dir: Path = typer.Option(Path("jobs"), "--jobs-dir"),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
    suite_id: str = typer.Option("default", "--suite-id"),
    trials_per_task: int = typer.Option(1, "--trials-per-task", min=1),
    concurrency: int = typer.Option(1, "--concurrency", min=1),
    seed: Optional[int] = typer.Option(None, "--seed"),
    resume: bool = typer.Option(False, "--resume"),
    replay_mode: str = typer.Option("off", "--replay-mode", case_sensitive=False),
    env: str = typer.Option(..., "--env", "-e", case_sensitive=False, help="Environment type (docker_cli, local_python, podman_cli, daytona)"),
    docker_image_tag_prefix: str = typer.Option("sentient-evals", "--docker-image-tag-prefix"),
    daytona_snapshot_template: Optional[str] = typer.Option(None, "--daytona-snapshot-template"),
    daytona_network_block_all: Optional[bool] = typer.Option(None, "--daytona-network-block-all"),
):
    """
    Run an eval suite from tasks JSON, task bundles directory, or registry dataset.
    """
    source_count = sum([tasks_path is not None, tasks_dir is not None, dataset is not None])
    if source_count != 1:
        raise typer.BadParameter("Provide exactly one of --tasks, --tasks-dir, or --dataset")
    if tasks_path is not None and env.lower() != EnvironmentType.local_python.value:
        raise typer.BadParameter("--env only supports local_python when using --tasks")
    if dataset is not None and registry_url is None and registry_path is None:
        raise typer.BadParameter("--dataset requires --registry-url or --registry-path")

    dataset_tasks_dir: Optional[Path] = None
    if dataset is not None:
        name, version = (dataset.split("@", 1) + [None])[:2]
        client = RegistryClientFactory.create(registry_url=registry_url, registry_path=registry_path)
        console.print(f"[cyan]Downloading dataset: {name}@{version or 'latest'}[/cyan]")
        downloaded = client.download_dataset(name, version)
        if not downloaded:
            raise typer.BadParameter(f"Dataset '{dataset}' has no tasks")
        dataset_tasks_dir = downloaded[0].local_path.parent
        console.print(f"[green]Downloaded {len(downloaded)} task(s) to {dataset_tasks_dir}[/green]")

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

    if agent_file is not None:
        try:
            ref = parse_agent_file_ref(agent_file)
            adapter_obj = load_agent_adapter_from_file(ref)
        except Exception as exc:
            raise typer.BadParameter(f"--agent-file invalid: {exc}") from exc
    elif adapter is not None:
        if adapter in _BUILTIN_ADAPTERS:
            adapter_type = adapter
            import_path = None
        else:
            adapter_type = "import"
            import_path = adapter
        kwargs = json.loads(adapter_kwargs) if adapter_kwargs else {}
        if model is not None:
            kwargs["model_name"] = model
        adapter_obj = build_adapter(adapter_type=adapter_type, import_path=import_path, kwargs=kwargs)
    elif spec is not None:
        adapter_obj = build_adapter(
            adapter_type=spec.adapter.type, import_path=spec.adapter.import_path, kwargs=spec.adapter.kwargs
        )
    else:
        raise typer.BadParameter("Provide --agent-file, --config, or --adapter to select an adapter")

    grader_specs: list[dict] = []
    if graders_file is not None:
        parsed = json.loads(graders_file.read_text(encoding="utf-8"))
        if not isinstance(parsed, list):
            raise typer.BadParameter("--graders-file must contain a JSON array of grader specs")
        grader_specs = parsed
    elif spec is not None:
        grader_specs = spec.graders
    elif tasks_dir is not None or dataset_tasks_dir is not None:
        effective_dir = tasks_dir or dataset_tasks_dir
        assert effective_dir is not None
        has_test_sh = any(effective_dir.rglob("tests/test.sh"))
        if not has_test_sh:
            raise typer.BadParameter(
                "No graders specified and no tests/test.sh found. "
                "Provide --config/--graders-file or add tests/test.sh verifiers."
            )
        grader_specs = [{"type": "verifier_script", "config": {}}]
    else:
        raise typer.BadParameter("Provide --graders-file or --config (graders) when using --tasks")

    graders = [build_grader(s) for s in grader_specs]
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S")

    jobs_dir = jobs_dir.expanduser().resolve()
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

    if tasks_dir is not None or dataset_tasks_dir is not None:
        effective_dir = tasks_dir or dataset_tasks_dir
        assert effective_dir is not None
        bundles = load_task_bundles(effective_dir)
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
    result_path = run_dir / "run_result.json"
    if not result_path.exists():
        raise typer.BadParameter("run_result.json not found")
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
    result_paths = sorted(trials_dir.glob("*/trial_result.json"))
    if not result_paths:
        raise typer.BadParameter("no trial_result.json files found under trials/")
    for result_path in result_paths:
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
    b_path = baseline_dir / "run_result.json"
    if not b_path.exists():
        raise typer.BadParameter("baseline run_result.json not found")
    c_path = candidate_dir / "run_result.json"
    if not c_path.exists():
        raise typer.BadParameter("candidate run_result.json not found")

    b = json.loads(b_path.read_text())
    c = json.loads(c_path.read_text())

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


@datasets_app.command("list")
def datasets_list(
    registry_url: Optional[str] = typer.Option(None, "--registry-url"),
    registry_path: Optional[Path] = typer.Option(None, "--registry-path", exists=True),
):
    """List all datasets available in a registry."""
    if registry_url is None and registry_path is None:
        raise typer.BadParameter("Provide --registry-url or --registry-path")

    client = RegistryClientFactory.create(
        registry_url=registry_url,
        registry_path=registry_path,
    )
    datasets = client.get_datasets()

    if not datasets:
        console.print("[yellow]No datasets found[/yellow]")
        return

    table = Table(title="Available Datasets", show_lines=True)
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="magenta")
    table.add_column("Tasks", style="green", justify="right")
    table.add_column("Description", style="white")

    for ds in sorted(datasets, key=lambda d: (d.name, d.version)):
        table.add_row(ds.name, ds.version, str(ds.task_count), ds.description[:60])

    console.print(table)
    console.print(f"\n[green]Total: {len(datasets)} dataset(s)[/green]")


@datasets_app.command("pull")
def datasets_pull(
    dataset: str = typer.Argument(..., help="Dataset in format 'name@version' or 'name'"),
    registry_url: Optional[str] = typer.Option(None, "--registry-url"),
    registry_path: Optional[Path] = typer.Option(None, "--registry-path", exists=True),
    output_dir: Optional[Path] = typer.Option(None, "-o", "--output-dir"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """Download a dataset from a registry."""
    if registry_url is None and registry_path is None:
        raise typer.BadParameter("Provide --registry-url or --registry-path")

    name, version = (dataset.split("@", 1) + [None])[:2]

    client = RegistryClientFactory.create(
        registry_url=registry_url,
        registry_path=registry_path,
    )

    console.print(f"[cyan]Downloading dataset: {name} (version: {version or 'latest'})[/cyan]")

    try:
        downloaded = client.download_dataset(
            name=name,
            version=version,
            output_dir=output_dir,
            overwrite=overwrite,
        )
        console.print(f"[green]Downloaded {len(downloaded)} task(s)[/green]")
        for task in downloaded[:5]:
            console.print(f"  {task.local_path}")
        if len(downloaded) > 5:
            console.print(f"  ... and {len(downloaded) - 5} more")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@datasets_app.command("path")
def datasets_path(
    dataset: str = typer.Argument(..., help="Dataset in format 'name@version' or 'name'"),
    registry_url: Optional[str] = typer.Option(None, "--registry-url"),
    registry_path: Optional[Path] = typer.Option(None, "--registry-path", exists=True),
):
    """Show the cached path for a downloaded dataset."""
    if registry_url is None and registry_path is None:
        raise typer.BadParameter("Provide --registry-url or --registry-path")

    name, version = (dataset.split("@", 1) + [None])[:2]

    client = DatasetClient(
        registry_url=registry_url,
        registry_path=registry_path,
    )

    path = client.get_dataset_path(name, version)
    if path is None:
        console.print(f"[yellow]Dataset '{dataset}' not cached. Run 'datasets pull' first.[/yellow]")
        raise typer.Exit(1)
    console.print(str(path.parent))


@tasks_app.command("create")
def tasks_create(
    name: str = typer.Argument(..., help="Name of the new task"),
    output_dir: Path = typer.Option(Path("."), "-o", "--output-dir", help="Directory to create task in"),
):
    """Scaffold a new task bundle with default structure."""
    task_dir = output_dir / name
    if task_dir.exists():
        console.print(f"[red]Error: Directory '{task_dir}' already exists[/red]")
        raise typer.Exit(1)

    task_dir.mkdir(parents=True)
    (task_dir / "environment").mkdir()
    (task_dir / "tests").mkdir()

    (task_dir / "instruction.md").write_text(
        f"# {name}\n\nDescribe the task instructions here.\n",
        encoding="utf-8",
    )

    (task_dir / "task.toml").write_text(
        f'[task]\nid = "{name}"\ntimeout = 300\n\n[metadata]\ndifficulty = "medium"\ntags = []\n',
        encoding="utf-8",
    )

    (task_dir / "environment" / "Dockerfile").write_text(
        "FROM python:3.11-slim\n\nWORKDIR /workspace\n\n# Add dependencies here\n",
        encoding="utf-8",
    )

    (task_dir / "tests" / "test.sh").write_text(
        '#!/bin/bash\nset -e\n\n# Add verification logic here\n# Exit 0 for pass, non-zero for fail\n\nexit 0\n',
        encoding="utf-8",
    )

    console.print(f"[green]Created task scaffold at: {task_dir}[/green]")
    console.print("Files created:")
    console.print(f"  {task_dir}/instruction.md")
    console.print(f"  {task_dir}/task.toml")
    console.print(f"  {task_dir}/environment/Dockerfile")
    console.print(f"  {task_dir}/tests/test.sh")
