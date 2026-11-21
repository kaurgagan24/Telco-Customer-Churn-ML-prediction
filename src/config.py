"""Config loading and path helpers."""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load the YAML config (defaults to configs/config.yaml)."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to the project root (absolute paths pass through)."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_parent(path: str | Path) -> Path:
    """Resolve a path and create its parent directory."""
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
