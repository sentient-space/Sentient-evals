from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable
import getpass
import contextlib
import subprocess
import shutil
from collections import Counter

import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm

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
from .datasets.registry.base import BaseRegistryClient
from .prompts import select_adapter, select_dataset, select_environment, prompt_model_name, prompt_github_token
from .config import (
    get_github_token,
    get_saved_github_token,
    save_github_token,
    clear_github_token,
    CONFIG_FILE,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

datasets_app = typer.Typer(no_args_is_help=True)
app.add_typer(datasets_app, name="datasets")

tasks_app = typer.Typer(no_args_is_help=True)
app.add_typer(tasks_app, name="tasks")

auth_app = typer.Typer(no_args_is_help=True)
app.add_typer(auth_app, name="auth")

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

_PROVIDER_ENV_HINTS = {
    "openai": (["OPENAI_API_KEY"], None),
    "anthropic": (["ANTHROPIC_API_KEY"], None),
    "google": (None, ["GEMINI_API_KEY", "GOOGLE_API_KEY"]),
    "gemini": (None, ["GEMINI_API_KEY", "GOOGLE_API_KEY"]),
    "groq": (["GROQ_API_KEY"], None),
    "mistral": (["MISTRAL_API_KEY"], None),
    "together": (["TOGETHER_API_KEY"], None),
    "deepseek": (["DEEPSEEK_API_KEY"], None),
    "xai": (["XAI_API_KEY"], None),
    "azure": (["AZURE_RESOURCE_NAME", "AZURE_API_KEY"], None),
    "databricks": (["DATABRICKS_HOST", "DATABRICKS_TOKEN"], None),
    "tetrate": (["TETRATE_API_KEY"], None),
}


def _prompt_secret_env(name: str, *, label: str | None = None) -> bool:
    if os.environ.get(name):
        return True
    if not sys.stdin.isatty():
        return False
    prompt = label or name
    if not Confirm.ask(f"{prompt} not set. Enter now?", default=False):
        return False
    value = getpass.getpass(f"Enter {prompt}: ").strip()
    if not value:
        return False
    os.environ[name] = value
    return True


def _prompt_any_of(env_vars: list[str], *, label: str | None = None) -> None:
    if any(os.environ.get(v) for v in env_vars):
        return
    for v in env_vars:
        if _prompt_secret_env(v, label=label or v):
            return


def _prompt_required(env_vars: list[str]) -> None:
    for v in env_vars:
        if not os.environ.get(v):
            _prompt_secret_env(v, label=v)


def _maybe_prompt_api_keys(adapter: str, model: str | None) -> None:
    if not sys.stdin.isatty():
        return

    if adapter == "codex":
        _prompt_secret_env("OPENAI_API_KEY")
        return
    if adapter == "cursor-cli":
        _prompt_secret_env("CURSOR_API_KEY")
        return
    if adapter == "cline-cli":
        _prompt_secret_env("API_KEY")
        return
    if adapter == "claude-code":
        _prompt_any_of(["ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"], label="Anthropic/Claude auth")
        return
    if adapter == "gemini-cli":
        _prompt_any_of(["GEMINI_API_KEY", "GOOGLE_API_KEY"], label="Gemini API key")
        return
    if adapter == "aider":
        _prompt_any_of(["OPENAI_API_KEY", "ANTHROPIC_API_KEY"], label="Aider provider API key")
        return
    if adapter == "qwen-coder":
        _prompt_secret_env("OPENAI_API_KEY")
        return
    if adapter == "openhands":
        _prompt_any_of(
            ["LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"],
            label="OpenHands API key",
        )
        return
    if adapter == "mini-swe-agent":
        _prompt_any_of(
            ["MSWEA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"],
            label="Mini SWE Agent API key",
        )
        return
    if adapter == "swe-agent":
        _prompt_any_of(["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TOGETHER_API_KEY"], label="SWE Agent API key")
        return

    if adapter in {"goose", "opencode"} and model and "/" in model:
        provider = model.split("/", 1)[0].lower()
        required, any_of = _PROVIDER_ENV_HINTS.get(provider, (None, None))
        if required:
            _prompt_required(required)
        elif any_of:
            _prompt_any_of(any_of, label=f"{provider} API key")


def _format_run_event(name: str, payload: dict[str, object]) -> str | None:
    trial_id = payload.get("trial_id")
    prefix = f"{trial_id}: " if trial_id else ""
    if name == "run_start":
        return f"Starting run {payload.get('run_id')} ({payload.get('trial_count')} trials)"
    if name == "env_start":
        return f"{prefix}setting up environment"
    if name == "env_ready":
        return f"{prefix}environment ready"
    if name == "adapter_start":
        return f"{prefix}running agent"
    if name == "adapter_done":
        return f"{prefix}agent completed"
    if name == "grading_start":
        return f"{prefix}grading"
    if name == "grading_done":
        return f"{prefix}grading completed"
    if name == "trial_error":
        return f"{prefix}failed"
    if name == "trial_done":
        ok = payload.get("ok")
        return f"{prefix}done ({'ok' if ok else 'failed'})"
    if name == "run_done":
        return "Finalizing results"
    return None


def _make_status_callback(status) -> Callable[[str, dict[str, object]], None]:
    def _on_event(name: str, payload: dict[str, object]) -> None:
        msg = _format_run_event(name, payload)
        if msg:
            status.update(msg)
    return _on_event


def _dataset_cache_dir(name: str, version: str | None) -> Path:
    safe_version = version or "head"
    return Path.home() / ".cache" / "sentient-evals" / "datasets" / f"{name}@{safe_version}"


def _cached_task_counts(tasks_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not tasks_dir.exists():
        return counts
    for p in tasks_dir.iterdir():
        if p.is_dir() and (p / "task.toml").exists():
            counts[p.name] += 1
    return counts


def _expected_task_counts(tasks) -> Counter[str]:
    counts: Counter[str] = Counter()
    for task in tasks:
        name = Path(task.path).name
        counts[name] += 1
    return counts


def _summarize_cache_mismatch(expected: Counter[str], cached: Counter[str]) -> dict[str, object]:
    missing = expected - cached
    extra = cached - expected

    def _sample(counter: Counter[str], limit: int = 5) -> list[str]:
        items: list[str] = []
        for name in sorted(counter.keys()):
            count = counter[name]
            items.append(f"{name} (x{count})" if count > 1 else name)
            if len(items) >= limit:
                break
        return items

    return {
        "expected_total": sum(expected.values()),
        "cached_total": sum(cached.values()),
        "missing_total": sum(missing.values()),
        "extra_total": sum(extra.values()),
        "missing_sample": _sample(missing),
        "extra_sample": _sample(extra),
    }


def _resolve_dataset_tasks_dir(
    *,
    name: str,
    version: str | None,
    client: BaseRegistryClient,
) -> Path:
    spec = client.get_dataset_spec(name, version)
    if not spec.tasks:
        raise typer.BadParameter(f"Dataset '{name}@{spec.version}' has no tasks")

    cache_dir = _dataset_cache_dir(spec.name, spec.version)
    expected = _expected_task_counts(spec.tasks)
    cached = _cached_task_counts(cache_dir)
    cached_count = sum(cached.values())

    if cached_count > 0:
        if cached != expected:
            summary = _summarize_cache_mismatch(expected, cached)
            console.print(
                "[yellow]Cached dataset is incomplete or out-of-date; re-downloading.[/yellow]"
            )
            console.print(
                f"[yellow]Expected {summary['expected_total']} tasks, found {summary['cached_total']} "
                f"(missing {summary['missing_total']}, extra {summary['extra_total']}).[/yellow]"
            )
            if summary["missing_sample"]:
                console.print(f"[yellow]Missing (sample): {', '.join(summary['missing_sample'])}[/yellow]")
            if summary["extra_sample"]:
                console.print(f"[yellow]Extra (sample): {', '.join(summary['extra_sample'])}[/yellow]")
            shutil.rmtree(cache_dir, ignore_errors=True)
            cached = Counter()
            cached_count = 0
        else:
            use_cache = True
            if sys.stdin.isatty():
                use_cache = Confirm.ask(
                    f"Dataset cached ({cached_count} tasks). Use cached copy?",
                    default=True,
                )
            if use_cache:
                console.print(f"[green]Using cached dataset at {cache_dir}[/green]")
                return cache_dir
            # Refresh: wipe and re-download
            shutil.rmtree(cache_dir, ignore_errors=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Downloading dataset: {spec.name}@{spec.version}[/cyan]")
    downloaded = client.download_dataset(
        name=spec.name,
        version=spec.version,
        output_dir=cache_dir,
        overwrite=True,
    )
    if not downloaded:
        raise typer.BadParameter(f"Dataset '{spec.name}@{spec.version}' has no tasks")
    cached = _cached_task_counts(cache_dir)
    if cached != expected:
        summary = _summarize_cache_mismatch(expected, cached)
        raise typer.BadParameter(
            "Downloaded dataset cache is incomplete. "
            f"Expected {summary['expected_total']} tasks, found {summary['cached_total']} "
            f"(missing {summary['missing_total']}, extra {summary['extra_total']})."
        )
    console.print(f"[green]Downloaded {len(downloaded)} task(s) to {cache_dir}[/green]")
    return cache_dir


def _cleanup_run_containers(run_dir: Path, env_type: EnvironmentType) -> None:
    if env_type not in (EnvironmentType.docker_cli, EnvironmentType.podman_cli, EnvironmentType.docker_sdk):
        return
    engine = "docker" if env_type in (EnvironmentType.docker_cli, EnvironmentType.docker_sdk) else "podman"
    names: set[str] = set()
    for p in run_dir.glob("trials/*/env_logs/container_name.txt"):
        try:
            name = p.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if name:
            names.add(name)
    for name in sorted(names):
        subprocess.run([engine, "rm", "-f", name], check=False, capture_output=True)

def _create_registry_client_for_run(
    *, registry_url: Optional[str], registry_path: Optional[Path]
) -> BaseRegistryClient:
    github_token = None
    if registry_url is None and registry_path is None:
        github_token = os.environ.get("GITHUB_TOKEN")
        if github_token is None and sys.stdin.isatty():
            github_token = prompt_github_token()

    return RegistryClientFactory.create(
        registry_url=registry_url,
        registry_path=registry_path,
        use_default=True,
        github_token=github_token,
    )


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
    env: Optional[str] = typer.Option(None, "--env", "-e", case_sensitive=False, help="Environment type (docker_cli, local_python, podman_cli, daytona)"),
    docker_image_tag_prefix: str = typer.Option("sentient-evals", "--docker-image-tag-prefix"),
    daytona_snapshot_template: Optional[str] = typer.Option(None, "--daytona-snapshot-template"),
    daytona_network_block_all: Optional[bool] = typer.Option(None, "--daytona-network-block-all"),
):
    """
    Run an eval suite from tasks JSON, task bundles directory, or registry dataset.
    """
    if adapter is None and agent_file is None and config is None:
        adapter = select_adapter()
        if adapter is None:
            raise typer.Abort()

    if model is None and adapter is not None and adapter in _BUILTIN_ADAPTERS:
        model = prompt_model_name(adapter)

    if adapter is not None and adapter in _BUILTIN_ADAPTERS:
        _maybe_prompt_api_keys(adapter, model)

    if env is None:
        env = select_environment()
        if env is None:
            raise typer.Abort()

    registry_client = None
    if tasks_path is None and tasks_dir is None:
        if dataset is None:
            registry_client = _create_registry_client_for_run(
                registry_url=registry_url, registry_path=registry_path
            )
            datasets = registry_client.get_datasets()
            selected = select_dataset(datasets)
            if selected is None:
                raise typer.Abort()
            dataset = selected.get_qualified_name()
        else:
            registry_client = _create_registry_client_for_run(
                registry_url=registry_url, registry_path=registry_path
            )

    source_count = sum([tasks_path is not None, tasks_dir is not None, dataset is not None])
    if source_count != 1:
        raise typer.BadParameter("Provide exactly one of --tasks, --tasks-dir, or --dataset")
    if tasks_path is not None and env.lower() != EnvironmentType.local_python.value:
        raise typer.BadParameter("--env only supports local_python when using --tasks")

    dataset_tasks_dir: Optional[Path] = None
    if dataset is not None:
        name, version = (dataset.split("@", 1) + [None])[:2]
        client = registry_client or _create_registry_client_for_run(
            registry_url=registry_url, registry_path=registry_path
        )
        dataset_tasks_dir = _resolve_dataset_tasks_dir(
            name=name,
            version=version,
            client=client,
        )

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
    bundles = None
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
        bundles = load_task_bundles(effective_dir)
        if not bundles:
            raise typer.BadParameter(f"No task bundles found under: {effective_dir}")
        missing_tests = [
            b.task.id
            for b in bundles
            if b.tests_dir is None or not (b.tests_dir / "test.sh").exists()
        ]
        if missing_tests:
            missing_list = ", ".join(missing_tests)
            raise typer.BadParameter(
                "No graders specified and some tasks are missing tests/test.sh. "
                f"Add verifiers or provide --config/--graders-file. Missing: {missing_list}"
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

    status_ctx = (
        console.status("Starting evals...", spinner="dots") if console.is_terminal else contextlib.nullcontext()
    )
    run_dir = jobs_dir / run_id
    try:
        with status_ctx as status:
            on_event = _make_status_callback(status) if status is not None else None
            if tasks_dir is not None or dataset_tasks_dir is not None:
                effective_dir = tasks_dir or dataset_tasks_dir
                assert effective_dir is not None
                if bundles is None:
                    bundles = load_task_bundles(effective_dir)
                results, summary = asyncio.run(
                    run_suite_bundles(
                        bundles=bundles, adapter=adapter_obj, graders=graders, cfg=cfg, on_event=on_event
                    )
                )
            else:
                assert tasks_path is not None
                tasks_raw = json.loads(tasks_path.read_text())
                tasks = [Task(**t) for t in tasks_raw]
                results, summary = asyncio.run(
                    run_suite(tasks=tasks, adapter=adapter_obj, graders=graders, cfg=cfg, on_event=on_event)
                )
    except KeyboardInterrupt:
        cancel_path = run_dir / "cancel.json"
        if not cancel_path.exists():
            cancel_path.write_text(
                json.dumps({"requested_at": datetime.now(timezone.utc).isoformat()}), encoding="utf-8"
            )
        _cleanup_run_containers(run_dir, env_type)
        console.print("\n[yellow]Run cancelled. Cleaned up containers (best effort).[/yellow]")
        raise typer.Exit(130)

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
    """List all datasets available in a registry (defaults to GitHub harbor-datasets)."""
    github_token = None
    if registry_url is None and registry_path is None:
        github_token = os.environ.get("GITHUB_TOKEN")
        if github_token is None and sys.stdin.isatty():
            github_token = prompt_github_token()

    client = RegistryClientFactory.create(
        registry_url=registry_url,
        registry_path=registry_path,
        use_default=True,
        github_token=github_token,
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

    has_unknown_counts = False
    for ds in sorted(datasets, key=lambda d: (d.name, d.version)):
        if ds.task_count == 0:
            task_str = "—"
            has_unknown_counts = True
        else:
            task_str = str(ds.task_count)
        table.add_row(ds.name, ds.version, task_str, ds.description[:60])

    console.print(table)
    console.print(f"\n[green]Total: {len(datasets)} dataset(s)[/green]")

    if has_unknown_counts and github_token is None:
        console.print(
            "\n[dim]Tip: Set GITHUB_TOKEN env var to see task counts "
            "(avoids rate limits)[/dim]"
        )


@datasets_app.command("pull")
def datasets_pull(
    dataset: Optional[str] = typer.Argument(None, help="Dataset in format 'name@version' or 'name'"),
    registry_url: Optional[str] = typer.Option(None, "--registry-url"),
    registry_path: Optional[Path] = typer.Option(None, "--registry-path", exists=True),
    output_dir: Optional[Path] = typer.Option(None, "-o", "--output-dir"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """Download a dataset from a registry (defaults to GitHub harbor-datasets)."""
    github_token = None
    if registry_url is None and registry_path is None:
        github_token = os.environ.get("GITHUB_TOKEN")
        if github_token is None and sys.stdin.isatty():
            github_token = prompt_github_token()

    client = RegistryClientFactory.create(
        registry_url=registry_url,
        registry_path=registry_path,
        use_default=True,
        github_token=github_token,
    )

    if dataset is None:
        datasets = client.get_datasets()
        selected = select_dataset(datasets)
        if selected is None:
            raise typer.Abort()
        dataset = selected.get_qualified_name()

    name, version = (dataset.split("@", 1) + [None])[:2]

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
        f'id = "{name}"\n'
        f'timeout_seconds = 300\n\n'
        f'[input]\n\n'
        f'[metadata]\n'
        f'difficulty = "medium"\n'
        f'tags = []\n\n'
        f'[environment]\n'
        f'type = "container"\n'
        f'allow_internet = true\n',
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



@auth_app.command("login")
def auth_login():
    """Save GitHub token for authenticated API access."""
   
    
    existing = get_saved_github_token()
    if existing:
        console.print(f"[yellow]Already logged in.[/yellow] Token saved in {CONFIG_FILE}")
        console.print("Run 'sentient-evals auth logout' to remove it first.")
        return
    
    console.print(
        "[bold]GitHub token setup[/bold]\n"
        "This token enables full API access (5000 requests/hour).\n"
        "Generate one at: https://github.com/settings/tokens\n"
        "Required scope: public_repo\n"
    )
    
    token = getpass.getpass("Paste your GitHub token: ")
    token = token.strip()
    
    if not token:
        console.print("[yellow]No token provided. Cancelled.[/yellow]")
        raise typer.Exit(1)
    
    if not token.startswith(("ghp_", "github_pat_")):
        console.print("[yellow]Warning: Token doesn't look like a GitHub token, but saving anyway.[/yellow]")
    
    save_github_token(token)
    console.print(f"[green]✓ Token saved to {CONFIG_FILE}[/green]")
    console.print("[dim]Token permissions: owner read/write only (chmod 600)[/dim]")


@auth_app.command("logout")
def auth_logout():
    """Remove saved GitHub token."""
    existing = get_saved_github_token()
    if not existing:
        if os.environ.get("GITHUB_TOKEN"):
            console.print("[yellow]No saved token found. GITHUB_TOKEN is set in the environment.[/yellow]")
            console.print("Unset GITHUB_TOKEN to fully log out.")
            return
        console.print("[yellow]Not logged in. No token to remove.[/yellow]")
        return
    
    clear_github_token()
    console.print(f"[green]✓ Token removed from {CONFIG_FILE}[/green]")


@auth_app.command("status")
def auth_status():
    """Show authentication status."""
    env_token = os.environ.get("GITHUB_TOKEN")
    saved_token = get_saved_github_token()
    
    if env_token:
        console.print("[green]✓ Authenticated via GITHUB_TOKEN environment variable[/green]")
        console.print(f"  Token: {env_token[:8]}...{env_token[-4:]}")
    if saved_token:
        console.print(f"[green]✓ Authenticated via saved token[/green]")
        console.print(f"  Config: {CONFIG_FILE}")
        console.print(f"  Token: {saved_token[:8]}...{saved_token[-4:]}")
    if not env_token and not saved_token:
        console.print("[yellow]✗ Not authenticated[/yellow]")
        console.print("  Run 'sentient-evals auth login' to authenticate")
        console.print("  Or set GITHUB_TOKEN environment variable")
