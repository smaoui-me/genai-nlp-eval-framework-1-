from pathlib import Path

from genai_eval.llm_client import call_llm, get_model_name
from genai_eval.extraction_methods.base import ExtractionMethod
from genai_eval.json_utils import parse_json_response
from genai_eval.prompts import format_prompt, load_prompt_template
from genai_eval.config_loader import load_config

DEFAULT_CONFIG_PATH = Path("configs/zero_shot.yaml")


class ZeroShotTicketExtraction(ExtractionMethod):
    """Zero-shot extraction: single prompt, no examples."""

    name = "zero_shot"

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config = load_config(config_path)
        self.prompt_template = load_prompt_template(Path(self.config["prompt_path"]))
        self.llm_params = self.config.get("llm", {})
        self.top_k_tags = self.config.get("extraction", {}).get("top_k_tags")
        self.model_name = get_model_name()

    def extract_record(
        self,
        text: str,
        allowed_labels: dict,
        context: dict | None = None,
    ) -> dict:
        candidate_tags = (context or {}).get("candidate_tags", allowed_labels.get("tags", []))
        prompt = format_prompt(
            self.prompt_template,
            ticket_text=text,
            allowed_types=allowed_labels["types"],
            allowed_queues=allowed_labels["queues"],
            candidate_tags=candidate_tags,
        )
        raw_response = call_llm(prompt, **self.llm_params)
        parsed_output, json_valid, json_error = parse_json_response(raw_response)
        validated_output, validation = self.validate_against_labels(
            prediction=parsed_output,
            allowed_labels=allowed_labels,
            candidate_tags=candidate_tags,
            top_k_tags=self.top_k_tags,
        )
        return {
            "raw_responses": {"step_single": raw_response},
            "parsed_output": parsed_output if isinstance(parsed_output, dict) else {},
            "validated_output": validated_output,
            "json_validity": {
                "step_single_json_valid": json_valid,
                "step_single_json_error": json_error,
                "all_json_valid": json_valid,
            },
            "validation": validation,
            "prompt_inputs": {"candidate_tags": candidate_tags},
        }

    def extract(self, text: str, allowed_labels: dict, context: dict | None = None) -> dict:
        return self.extract_record(text, allowed_labels, context=context)["validated_output"]
