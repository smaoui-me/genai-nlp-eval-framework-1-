"""Load reusable extraction/routing use-case templates from YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config" / "use_cases"


def load_use_cases(directory: Path = CONFIG_DIR) -> dict[str, dict]:
    """Return validated templates keyed by filename stem."""
    templates: dict[str, dict] = {}
    for path in sorted(directory.glob("*.yaml")):
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(value, dict):
            raise ValueError(f"Use-case config must be an object: {path}")
        missing = {"name", "entity_labels"} - set(value)
        if missing:
            raise ValueError(f"Use-case config {path} is missing: {sorted(missing)}")
        labels = value["entity_labels"]
        if not isinstance(labels, list) or not labels or not all(str(label).strip() for label in labels):
            raise ValueError(f"Use-case config {path} needs a non-empty entity_labels list")
        value["entity_labels"] = list(dict.fromkeys(str(label).strip() for label in labels))
        definitions = value.get("label_definitions", {})
        if not isinstance(definitions, dict):
            raise ValueError(f"label_definitions must be an object: {path}")
        routing = value.setdefault("routing", {})
        if not isinstance(routing, dict):
            raise ValueError(f"routing must be an object: {path}")
        routing["departments"] = list(dict.fromkeys(
            str(item).strip() for item in routing.get("departments", []) if str(item).strip()
        ))
        value["id"] = path.stem
        value["path"] = str(path)
        templates[path.stem] = value
    if not templates:
        raise RuntimeError(f"No use-case templates found in {directory}")
    return templates
