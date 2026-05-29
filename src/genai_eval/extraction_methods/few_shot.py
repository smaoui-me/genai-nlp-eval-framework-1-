"""
few_shot.py

Few-shot ticket extraction. Identical to zero-shot but uses the few-shot
prompt template and its own config file.
"""

from pathlib import Path

from genai_eval.llm_client import call_llm, get_model_name
from genai_eval.extraction_methods.zero_shot import ZeroShotTicketExtraction
from genai_eval.config_loader import load_config
from genai_eval.json_utils import parse_json_response
from genai_eval.prompts import format_prompt, load_prompt_template

DEFAULT_CONFIG_PATH = Path("configs/few_shot.yaml")


class FewShotTicketExtraction(ZeroShotTicketExtraction):
    """Few-shot extraction: prompt with 3 demonstration examples."""

    name = "few_shot"

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config = load_config(config_path)
        self.prompt_template = load_prompt_template(
            Path(self.config["prompt_path"])
        )
        self.llm_params = self.config.get("llm", {})
        self.top_k_tags = self.config.get("extraction", {}).get("top_k_tags")
        self.model_name = get_model_name()

    def extract(self, text: str, allowed_labels: dict, context: dict | None = None) -> dict:
        """Extract type, queue, and tags using few-shot prompting."""
        candidate_tags = (context or {}).get("candidate_tags", allowed_labels.get("tags", []))
        prompt = format_prompt(
            self.prompt_template,
            ticket_text=text,
            allowed_types=allowed_labels["types"],
            allowed_queues=allowed_labels["queues"],
            candidate_tags=candidate_tags,
        )
        raw_response = call_llm(prompt, **self.llm_params)
        prediction, _, _ = parse_json_response(raw_response)
        validated_output, _ = self.validate_against_labels(
            prediction=prediction,
            allowed_labels=allowed_labels,
            candidate_tags=candidate_tags,
            top_k_tags=self.top_k_tags,
        )
        return validated_output
