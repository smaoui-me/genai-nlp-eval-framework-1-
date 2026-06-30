"""Few-shot extraction method for FewNERD."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from genai_eval.config_loader import load_config
from genai_eval.extraction.labels import FINE_LABELS
from genai_eval.extraction.methods.base import ExtractionMethod
from genai_eval.extraction.utils import build_spans, parse_int_array, parse_token_array, sentence_from_tokens
from genai_eval.json_utils import parse_json_response
from genai_eval.llm_client import call_llm, get_model_name
from genai_eval.prompts import format_prompt, load_prompt_template

DEFAULT_CONFIG_PATH = Path("configs/extraction/few_shot.yaml")


def _format_examples(examples: list[dict]) -> str:
    parts = []
    for ex in examples:
        sentence = sentence_from_tokens(ex["tokens"])
        output = json.dumps({"entities": ex["spans"]}, ensure_ascii=False)
        parts.append(f"Sentence:\n{sentence}\n\nOutput:\n{output}")
    return "\n\n---\n\n".join(parts)


class FewShotExtractionMethod(ExtractionMethod):
    name = "few_shot"

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config = load_config(config_path)
        self.prompt_template = load_prompt_template(Path(self.config["prompt_path"]))
        self.llm_params = self.config.get("llm", {})
        self.retry_on_malformed = int(self.config.get("runtime", {}).get("retry_on_malformed_json", 1))
        self.model_name = get_model_name()
        few_shot_cfg = self.config.get("few_shot", {})
        self.n_examples = few_shot_cfg.get("n_examples", 3)
        self.seed = few_shot_cfg.get("seed", 42)
        self.examples_path = Path(few_shot_cfg.get("examples_path", "data/extraction/raw/intra/train-00000-of-00001.csv"))
        self._examples: list[dict] | None = None

    @property
    def examples(self) -> list[dict]:
        if self._examples is None:
            self._examples = self._load_examples()
        return self._examples

    def _load_examples(self) -> list[dict]:
        df = pd.read_csv(self.examples_path)
        sampled = df.sample(n=min(self.n_examples, len(df)), random_state=self.seed)
        result = []
        for _, row in sampled.iterrows():
            tokens = parse_token_array(row["tokens"])
            fine_tags = parse_int_array(row["fine_ner_tags"])
            spans = build_spans(tokens, fine_tags, FINE_LABELS)
            result.append({"tokens": tokens, "spans": spans})
        return result

    def extract_record(self, tokens: list[str], allowed_entity_types: list[str]) -> dict:
        text = sentence_from_tokens(tokens)
        examples_text = _format_examples(self.examples)
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
                examples=examples_text,
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
