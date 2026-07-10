from __future__ import annotations

import json
from pathlib import Path

from genai_eval.config_loader import load_config
from genai_eval.extraction.methods.base import ExtractionMethod
from genai_eval.json_utils import parse_json_response
from genai_eval.llm_client import call_llm, get_model_name
from genai_eval.prompts import format_prompt, load_prompt_template

DEFAULT_CONFIG_PATH = Path("configs/extraction/agent_verify_extraction.yaml")


class AgentVerifyExtractionMethod(ExtractionMethod):
    name = "agent_verify_extraction"

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config = load_config(config_path)
        self.step1_prompt_template = load_prompt_template(Path(self.config["prompt_path"]))
        self.step2_prompt_template = load_prompt_template(Path(self.config["verification_prompt_path"]))
        self.llm_params = self.config.get("llm", {})
        self.retry_on_malformed = int(self.config.get("runtime", {}).get("retry_on_malformed_json", 1))
        self.model_name = get_model_name()

    def _call_structured_prompt(self, prompt_template: str, sentence: str, indexed_tokens: str, step1_json: str | None = None):
        raw_responses = {}
        parsed_output = {}
        json_valid = False
        json_error = None
        attempts = self.retry_on_malformed + 1

        for attempt in range(1, attempts + 1):
            prompt_kwargs = {
                "sentence": sentence,
                "indexed_tokens": indexed_tokens,
            }
            if step1_json is not None:
                prompt_kwargs["step1_json"] = step1_json
            prompt = format_prompt(prompt_template, **prompt_kwargs)
            raw_response = call_llm(prompt, **self.llm_params)
            raw_responses[f"attempt_{attempt}"] = raw_response
            parsed_output, json_valid, json_error = parse_json_response(raw_response)
            if json_valid:
                break

        return raw_responses, parsed_output, json_valid, json_error

    @staticmethod
    def _normalize_prediction(parsed_output: dict | None) -> dict:
        entities = parsed_output.get("entities", []) if isinstance(parsed_output, dict) else []
        return {
            "entities": [
                {
                    "text": entity.get("text", ""),
                    "type": "location",
                    "start": entity.get("start"),
                    "end": entity.get("end"),
                }
                for entity in entities
                if isinstance(entity, dict)
            ]
        }

    @staticmethod
    def _entity_key(entity: dict) -> tuple:
        return (entity.get("text", ""), entity.get("start"), entity.get("end"))

    def extract_record(self, sentence: str, tokens: list[str], allowed_entity_types: list[str]) -> dict:
        indexed_tokens = "\n".join(f"{index}: {token}" for index, token in enumerate(tokens))

        step1_raw, step1_parsed, step1_json_valid, step1_json_error = self._call_structured_prompt(
            self.step1_prompt_template,
            sentence=sentence,
            indexed_tokens=indexed_tokens,
        )
        step1_normalized = self._normalize_prediction(step1_parsed if isinstance(step1_parsed, dict) else {})
        step1_validated, step1_validation = self.validate_against_labels(
            prediction=step1_normalized,
            allowed_entity_types=allowed_entity_types,
            tokens=tokens,
        )

        step1_json = json.dumps(
            {"entities": [{"text": entity["text"], "start": entity["start"], "end": entity["end"]} for entity in step1_validated["entities"]]},
            ensure_ascii=False,
        )

        step2_raw, step2_parsed, step2_json_valid, step2_json_error = self._call_structured_prompt(
            self.step2_prompt_template,
            sentence=sentence,
            indexed_tokens=indexed_tokens,
            step1_json=step1_json,
        )
        step2_normalized = self._normalize_prediction(step2_parsed if isinstance(step2_parsed, dict) else {})
        step2_validated, step2_validation = self.validate_against_labels(
            prediction=step2_normalized,
            allowed_entity_types=allowed_entity_types,
            tokens=tokens,
        )

        if step2_json_valid:
            final_validated = step2_validated
            final_validation = step2_validation
        else:
            final_validated = step1_validated
            final_validation = step1_validation

        step1_keys = {self._entity_key(entity) for entity in step1_validated["entities"]}
        final_keys = {self._entity_key(entity) for entity in final_validated["entities"]}
        verification_debug = {
            "step1": {
                "raw_responses": step1_raw,
                "parsed_output": step1_parsed if isinstance(step1_parsed, dict) else {},
                "validated_output": step1_validated,
                "json_valid": step1_json_valid,
                "json_error": step1_json_error,
                "validation": step1_validation,
            },
            "step2": {
                "raw_responses": step2_raw,
                "parsed_output": step2_parsed if isinstance(step2_parsed, dict) else {},
                "validated_output": step2_validated,
                "json_valid": step2_json_valid,
                "json_error": step2_json_error,
                "validation": step2_validation,
                "used_for_scoring": step2_json_valid,
                "fallback_to_step1": not step2_json_valid,
            },
            "diff": {
                "added": [entity for entity in final_validated["entities"] if self._entity_key(entity) not in step1_keys],
                "removed": [entity for entity in step1_validated["entities"] if self._entity_key(entity) not in final_keys],
                "rephrased_or_shifted": [
                    {"step1": step1_entity, "final": final_entity}
                    for step1_entity in step1_validated["entities"]
                    for final_entity in final_validated["entities"]
                    if step1_entity.get("text") == final_entity.get("text")
                    and self._entity_key(step1_entity) != self._entity_key(final_entity)
                ],
            },
        }

        return {
            "raw_responses": {
                "step1": step1_raw,
                "step2": step2_raw,
            },
            "parsed_output": {
                "step1": step1_parsed if isinstance(step1_parsed, dict) else {},
                "step2": step2_parsed if isinstance(step2_parsed, dict) else {},
                "final": {"entities": [{"text": entity["text"], "start": entity["start"], "end": entity["end"]} for entity in final_validated["entities"]]},
            },
            "validated_output": final_validated,
            "json_validity": {
                "step1_json_valid": step1_json_valid,
                "step1_json_error": step1_json_error,
                "step2_json_valid": step2_json_valid,
                "step2_json_error": step2_json_error,
                "all_json_valid": step2_json_valid,
                "used_step1_fallback": not step2_json_valid,
            },
            "validation": final_validation,
            "verification_debug": verification_debug,
        }
