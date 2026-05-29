"""
zero_shot.py

Zero-shot ticket extraction using a single LLM prompt.
Config is loaded from configs/zero_shot.yaml.
"""

import json
import re
from pathlib import Path

import yaml

from genai_eval.llm_client import call_llm
from genai_eval.extraction_methods.base import ExtractionMethod

DEFAULT_CONFIG_PATH = Path("configs/zero_shot.yaml")


def load_config(path: Path) -> dict:
    """Load YAML config from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompt_template(path: Path) -> str:
    """Load the prompt template from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def format_label_list(labels: list) -> str:
    """Format a list of labels as a bulleted string for the prompt."""
    return "\n".join(f"- {label}" for label in labels)


def format_prompt(template: str, text: str, allowed_labels: dict) -> str:
    """Fill the prompt template with ticket text and allowed labels."""
    return template.format(
        ticket_text=text,
        allowed_types=format_label_list(allowed_labels["types"]),
        allowed_queues=format_label_list(allowed_labels["queues"]),
        allowed_tags=format_label_list(allowed_labels["tags"]),
    )


def parse_llm_response(response: str) -> dict:
    """Strip markdown fences and parse JSON from the LLM response."""
    cleaned = re.sub(r"```(?:json)?", "", response).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse LLM response as JSON: {e}\nResponse: {response!r}"
        )


class ZeroShotTicketExtraction(ExtractionMethod):
    """Zero-shot extraction: single prompt, no examples."""

    name = "zero_shot"

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config = load_config(config_path)
        self.prompt_template = load_prompt_template(
            Path(self.config["prompt_path"])
        )
        self.llm_params = self.config.get("llm", {})

    def extract(self, text: str, allowed_labels: dict) -> dict:
        """Extract type, queue, and tags using zero-shot prompting.

        Args:
            text: Ticket input text.
            allowed_labels: Dict with keys "types", "queues", "tags".

        Returns:
            Standardized prediction dict.
        """
        prompt = format_prompt(self.prompt_template, text, allowed_labels)
        raw_response = call_llm(prompt, **self.llm_params)
        prediction = parse_llm_response(raw_response)
        self.validate_prediction(prediction)
        return prediction