from __future__ import annotations

from pathlib import Path

from genai_eval.config_loader import load_config
from genai_eval.extraction.methods.base import ExtractionMethod
from genai_eval.extraction.utils import parse_freeform_entities
from genai_eval.llm_client import call_llm, get_model_name
from genai_eval.prompts import format_prompt, load_prompt_template

DEFAULT_CONFIG_PATH = Path("configs/extraction/zero_shot_freeform.yaml")


class ZeroShotFreeformExtractionMethod(ExtractionMethod):
    name = "zero_shot_freeform"

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config = load_config(config_path)
        self.prompt_template = load_prompt_template(Path(self.config["prompt_path"]))
        self.llm_params = self.config.get("llm", {})
        self.model_name = get_model_name()

    def extract_record(self, sentence: str, tokens: list[str], allowed_entity_types: list[str]) -> dict:
        indexed_tokens = "\n".join(f"{index}: {token}" for index, token in enumerate(tokens))
        prompt = format_prompt(self.prompt_template, sentence=sentence, indexed_tokens=indexed_tokens)
        raw_response = call_llm(prompt, **self.llm_params)
        entities, malformed_lines = parse_freeform_entities(raw_response)
        normalized_prediction = {
            "entities": [
                {"text": entity["text"], "type": "location", "start": entity["start"], "end": entity["end"]}
                for entity in entities
            ]
        }
        validated_output, validation = self.validate_against_labels(
            prediction=normalized_prediction,
            allowed_entity_types=allowed_entity_types,
            tokens=tokens,
        )
        if malformed_lines:
            validation["has_invalid_labels"] = True
            validation.setdefault("invalid_entities", []).extend(
                {"entity": item["line"], "reason": item["reason"]} for item in malformed_lines
            )
        return {
            "raw_responses": {"attempt_1": raw_response},
            "parsed_output": {"entities": entities},
            "validated_output": validated_output,
            "json_validity": {
                "json_valid": None,
                "json_error": None,
                "all_json_valid": False,
                "not_applicable": True,
            },
            "validation": validation,
        }
