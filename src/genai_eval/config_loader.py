"""
config_loader.py

Loads a YAML config file and returns it as a dict.
"""

import yaml
from pathlib import Path


def load_config(path: str | Path) -> dict:
    """Load a YAML config file from disk.

    Args:
        path: Path to the .yaml config file.

    Returns:
        Parsed config as a dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)