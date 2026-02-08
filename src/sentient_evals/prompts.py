
from __future__ import annotations

from typing import TYPE_CHECKING
import os

from rich.console import Console
from rich.prompt import Prompt
import getpass

if TYPE_CHECKING:
    from .datasets.models import DatasetSpec

console = Console()

try:
    from InquirerPy import inquirer
    from InquirerPy.base.control import Choice

    INQUIRER_AVAILABLE = True
except ImportError:
    INQUIRER_AVAILABLE = False


_BUILTIN_ADAPTERS = [
    "claude-code",
    "cursor-cli",
    "gemini-cli",
    "codex",
    "aider",
    "cline-cli",
    "goose",
    "mini-swe-agent",
    "opencode",
    "openhands",
    "qwen-coder",
    "swe-agent",
]

_ENVIRONMENT_TYPES = [
    "docker_cli",
    "local_python",
    "docker_sdk",
    "podman_cli",
    "daytona",
]

_DEFAULT_MODELS = {
    "claude-code": "anthropic/claude-sonnet-4-20250514",
    "cursor-cli": "cursor/auto",
    "gemini-cli": "google/gemini-2.0-flash",
    "codex": "openai/codex-mini-latest",
    "aider": "anthropic/claude-sonnet-4-20250514",
    "goose": "anthropic/claude-sonnet-4-20250514",
    "openhands": "anthropic/claude-sonnet-4-20250514",
}


def select_dataset(datasets: list["DatasetSpec"]) -> "DatasetSpec | None":
    if not datasets:
        console.print("[yellow]No datasets available[/yellow]")
        return None

    if INQUIRER_AVAILABLE:
        choices = [
            Choice(
                value=ds,
                name=f"{ds.name} ({ds.task_count} tasks) - {ds.description[:40]}..."
                if len(ds.description) > 40
                else f"{ds.name} ({ds.task_count} tasks) - {ds.description}",
            )
            for ds in sorted(datasets, key=lambda d: d.name)
        ]
        
        try:
            result = inquirer.select(
                message="Select a dataset:",
                choices=choices,
                max_height="70%",
            ).execute()
            return result
        except KeyboardInterrupt:
            return None
    else:
        console.print("\n[bold]Available datasets:[/bold]")
        sorted_datasets = sorted(datasets, key=lambda d: d.name)
        for i, ds in enumerate(sorted_datasets, 1):
            console.print(f"  {i}. {ds.name} ({ds.task_count} tasks)")
        
        try:
            choice = Prompt.ask("Enter dataset number", default="1")
            idx = int(choice) - 1
            if 0 <= idx < len(sorted_datasets):
                return sorted_datasets[idx]
            console.print("[red]Invalid selection[/red]")
            return None
        except (ValueError, KeyboardInterrupt):
            return None


def select_adapter() -> str | None:
    """Interactively select an adapter from built-in list."""
    if INQUIRER_AVAILABLE:
        try:
            result = inquirer.select(
                message="Select an agent adapter:",
                choices=_BUILTIN_ADAPTERS,
            ).execute()
            return result
        except KeyboardInterrupt:
            return None
    else:
        console.print("\n[bold]Available adapters:[/bold]")
        for i, adapter in enumerate(_BUILTIN_ADAPTERS, 1):
            console.print(f"  {i}. {adapter}")
        
        try:
            choice = Prompt.ask("Enter adapter number", default="1")
            idx = int(choice) - 1
            if 0 <= idx < len(_BUILTIN_ADAPTERS):
                return _BUILTIN_ADAPTERS[idx]
            console.print("[red]Invalid selection[/red]")
            return None
        except (ValueError, KeyboardInterrupt):
            return None


def select_environment() -> str | None:
    """Interactively select an environment type."""
    if INQUIRER_AVAILABLE:
        try:
            result = inquirer.select(
                message="Select environment:",
                choices=_ENVIRONMENT_TYPES,
                default="docker_cli",
            ).execute()
            return result
        except KeyboardInterrupt:
            return None
    else:
        console.print("\n[bold]Available environments:[/bold]")
        for i, env in enumerate(_ENVIRONMENT_TYPES, 1):
            console.print(f"  {i}. {env}")
        
        try:
            choice = Prompt.ask("Enter environment number", default="1")
            idx = int(choice) - 1
            if 0 <= idx < len(_ENVIRONMENT_TYPES):
                return _ENVIRONMENT_TYPES[idx]
            console.print("[red]Invalid selection[/red]")
            return None
        except (ValueError, KeyboardInterrupt):
            return None


def prompt_model_name(adapter: str) -> str | None:
    """Prompt for a model name, with sensible defaults per adapter."""
    default = _DEFAULT_MODELS.get(adapter, "")
    hint = f" (default: {default})" if default else ""
    
    try:
        if INQUIRER_AVAILABLE:
            result = inquirer.text(
                message=f"Enter model name{hint}:",
                default=default,
            ).execute()
            return result if result else None
        else:
            result = Prompt.ask(f"Enter model name{hint}", default=default)
            return result if result else None
    except KeyboardInterrupt:
        return None


def prompt_github_token() -> str | None:

    from .config import get_saved_github_token, save_github_token, CONFIG_FILE
    
    saved_token = get_saved_github_token()
    if saved_token:
        console.print(f"[dim]Using saved GitHub token from {CONFIG_FILE}[/dim]")
        return saved_token

    env_token = os.environ.get("GITHUB_TOKEN")
    if env_token:
        console.print("[dim]Using GitHub token from GITHUB_TOKEN environment variable[/dim]")
        return env_token
    
    try:
        console.print(
            "[yellow]GitHub API rate limit:[/yellow] "
            "Paste token for full task counts, or press Enter to skip"
        )
        
        if INQUIRER_AVAILABLE:
            token = inquirer.secret(
                message="GitHub token:",
                transformer=lambda x: "••••••••" if x else "(skipped)",
            ).execute()
        else:
            token = getpass.getpass(prompt="GitHub token: ")
        
        token = token.strip() if token else None
        
        if token:
            console.print("[green]✓ Token accepted[/green]")
            
            if INQUIRER_AVAILABLE:
                save_it = inquirer.confirm(
                    message="Save token for future use?",
                    default=True,
                ).execute()
            else:
                from rich.prompt import Confirm
                save_it = Confirm.ask("Save token for future use?", default=True)
            
            if save_it:
                save_github_token(token)
                console.print(f"[dim]Saved to {CONFIG_FILE}[/dim]")
        
        return token
    except KeyboardInterrupt:
        console.print()
        return None


def prompt_concurrency(default: int = 1) -> int:
    """Prompt for concurrency with a sensible default."""
    default = max(1, int(default))
    try:
        if INQUIRER_AVAILABLE:
            try:
                result = inquirer.number(
                    message="Enter concurrency:",
                    default=default,
                    float_allowed=False,
                    min_allowed=1,
                ).execute()
                return int(result) if result else default
            except Exception:
                pass
        result = Prompt.ask("Enter concurrency", default=str(default))
        value = int(result)
        return value if value >= 1 else default
    except KeyboardInterrupt:
        return default
    except Exception:
        return default
