from __future__ import annotations

from pathlib import Path

from genai_eval.config_loader import load_config
from genai_eval.extraction.methods.base import ExtractionMethod
from genai_eval.extraction.utils import sentence_from_tokens
from genai_eval.json_utils import parse_json_response
from genai_eval.llm_client import call_llm, get_model_name
from genai_eval.prompts import format_prompt, load_prompt_template

DEFAULT_CONFIG_PATH = Path("configs/extraction/zero_shot.yaml")


class ZeroShotExtractionMethod(ExtractionMethod):
    name = "zero_shot"

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config = load_config(config_path)
        self.prompt_template = load_prompt_template(Path(self.config["prompt_path"]))
        self.llm_params = self.config.get("llm", {})
        self.retry_on_malformed = int(self.config.get("runtime", {}).get("retry_on_malformed_json", 1))
        self.model_name = get_model_name()

    def extract_record(self, tokens: list[str], allowed_entity_types: list[str]) -> dict:
        text = sentence_from_tokens(tokens)
        raw_responses = {}
        parsed_output = {}
        json_valid = False
        json_error = None

        attempts = self.retry_on_malformed + 1
        for attempt in range(1, attempts + 1):
            prompt = format_prompt(
                self.prompt_template,
                sentence=text,
                allowed_entity_types=allowed_entity_types,
            )
            raw_response = call_llm(prompt, **self.llm_params)
            raw_responses[f"attempt_{attempt}"] = raw_response
            parsed_output, json_valid, json_error = parse_json_response(raw_response)
            if json_valid:
                break

        validated_output, validation = self.validate_against_labels(
            prediction=parsed_output if isinstance(parsed_output, dict) else {},
            allowed_entity_types=allowed_entity_types,
            tokens=tokens,
        )

        return {
            "raw_responses": raw_responses,
            "parsed_output": parsed_output if isinstance(parsed_output, dict) else {},
            "validated_output": validated_output,
            "json_validity": {
                "json_valid": json_valid,
                "json_error": json_error,
                "all_json_valid": json_valid,
            },
            "validation": validation,
        }
