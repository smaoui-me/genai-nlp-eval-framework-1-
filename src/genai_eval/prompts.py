"""
Shared prompt-loading and formatting helpers.
"""

from pathlib import Path


def load_prompt_template(path: str | Path) -> str:
    """Load a prompt template from disk."""
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def format_label_list(labels: list[str]) -> str:
    """Format labels as a prompt-friendly bullet list."""
    return "\n".join(f"- {label}" for label in labels)


def format_prompt(template: str, **kwargs) -> str:
    """Fill a prompt template using named placeholders."""
    formatted_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, list):
            formatted_kwargs[key] = format_label_list(value)
        else:
            formatted_kwargs[key] = value
    return template.format(**formatted_kwargs)
