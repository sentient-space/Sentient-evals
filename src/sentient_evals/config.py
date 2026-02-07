

from __future__ import annotations

import os
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  

# Config directory following XDG Base Directory spec
CONFIG_DIR = Path.home() / ".config" / "sentient-evals"
CONFIG_FILE = CONFIG_DIR / "config.toml"


def get_config_dir() -> Path:
    """Get the config directory, creating if needed."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def load_config() -> dict:
    """Load config from disk, returns empty dict if not found."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        return tomllib.loads(CONFIG_FILE.read_text())
    except Exception:
        return {}


def save_config(config: dict) -> None:
    """Save config to disk in TOML format."""
    get_config_dir()
    
    lines = []
    for key, value in config.items():
        if isinstance(value, str):
            lines.append(f'{key} = "{value}"')
        elif isinstance(value, bool):
            lines.append(f'{key} = {"true" if value else "false"}')
        elif isinstance(value, (int, float)):
            lines.append(f'{key} = {value}')
    
    CONFIG_FILE.write_text("\n".join(lines) + "\n")
    CONFIG_FILE.chmod(0o600)


def get_github_token() -> str | None:
    """
    Get GitHub token from environment or config file.
    
    Priority:
    1. GITHUB_TOKEN environment variable
    2. Stored in config file
    
    Returns:
        Token string or None if not configured
    """
    env_token = os.environ.get("GITHUB_TOKEN")
    if env_token:
        return env_token
    return get_saved_github_token()


def get_saved_github_token() -> str | None:
    """Get GitHub token stored in the config file (ignores environment)."""
    config = load_config()
    return config.get("github_token")


def save_github_token(token: str) -> None:
    """Save GitHub token to config file."""
    config = load_config()
    config["github_token"] = token
    save_config(config)


def clear_github_token() -> None:
    """Remove saved GitHub token from config."""
    config = load_config()
    config.pop("github_token", None)
    save_config(config)
