from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from genai_eval.config_loader import load_config
from genai_eval.extraction.labels import COARSE_LABELS
from genai_eval.extraction.methods.base import ExtractionMethod
from genai_eval.extraction.utils import (
    build_location_spans,
    format_indexed_tokens,
    parse_int_array,
    parse_token_array,
    sentence_from_tokens,
)
from genai_eval.json_utils import parse_json_response
from genai_eval.llm_client import call_llm, get_model_name
from genai_eval.prompts import format_prompt, load_prompt_template

DEFAULT_CONFIG_PATH = Path("configs/extraction/few_shot_structured.yaml")
LOCATION_TAG_ID = next(tag_id for tag_id, label in COARSE_LABELS.items() if label == "location")


def _format_examples(examples: list[dict]) -> str:
    parts = []
    for ex in examples:
        indexed = format_indexed_tokens(ex["tokens"])
        sentence = sentence_from_tokens(ex["tokens"])
        output = json.dumps({"entities": ex["gold_entities"]}, ensure_ascii=False)
        parts.append(f"Sentence: {sentence}\n\nToken indices:\n{indexed}\n\nOutput: {output}")
    return "\n\n---\n\n".join(parts)


class FewShotStructuredExtractionMethod(ExtractionMethod):
    name = "few_shot_structured"

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config = load_config(config_path)
        self.prompt_template = load_prompt_template(Path(self.config["prompt_path"]))
        self.llm_params = self.config.get("llm", {})
        self.retry_on_malformed = int(self.config.get("runtime", {}).get("retry_on_malformed_json", 1))
        self.model_name = get_model_name()
        few_shot_cfg = self.config.get("few_shot", {})
        self.n_examples = few_shot_cfg.get("n_examples", 3)
        self.examples_path = Path(few_shot_cfg.get("examples_path", "data/extraction/raw/intra/train-00000-of-00001.csv"))
        self._examples: list[dict] | None = None

    @property
    def examples(self) -> list[dict]:
        if self._examples is None:
            self._examples = self._load_examples()
        return self._examples

    def _load_examples(self) -> list[dict]:
        df = pd.read_csv(self.examples_path, nrows=1000)
        result = []
        for _, row in df.iterrows():
            if len(result) >= self.n_examples:
                break
            tokens = parse_token_array(row["tokens"])
            coarse_tags = parse_int_array(row["ner_tags"])
            if len(tokens) != len(coarse_tags):
                continue
            gold_entities = build_location_spans(tokens, coarse_tags, LOCATION_TAG_ID)
            if not gold_entities:
                continue
            result.append({"tokens": tokens, "gold_entities": gold_entities})
        return result

    def extract_record(self, sentence: str, tokens: list[str], allowed_entity_types: list[str]) -> dict:
        indexed_tokens = format_indexed_tokens(tokens)
        examples_text = _format_examples(self.examples)
        raw_responses = {}
        parsed_output = {}
        json_valid = False
        json_error = None

        attempts = self.retry_on_malformed + 1
        for attempt in range(1, attempts + 1):
            prompt = format_prompt(
                self.prompt_template,
                sentence=sentence,
                indexed_tokens=indexed_tokens,
                examples=examples_text,
            )
            raw_response = call_llm(prompt, **self.llm_params)
            raw_responses[f"attempt_{attempt}"] = raw_response
            parsed_output, json_valid, json_error = parse_json_response(raw_response)
            if json_valid:
                break

        if isinstance(parsed_output, dict):
            entities = parsed_output.get("entities", [])
        else:
            entities = []
        normalized_prediction = {
            "entities": [
                {"text": entity.get("text", ""), "type": "location", "start": entity.get("start"), "end": entity.get("end")}
                for entity in entities
                if isinstance(entity, dict)
            ]
        }
        validated_output, validation = self.validate_against_labels(
            prediction=normalized_prediction,
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
