"""
Base class for ticket classification with evidence.
"""

from abc import ABC, abstractmethod

from genai_eval.prediction_validation import validate_and_clean_prediction


class EvidenceClassificationMethod(ABC):
    name: str = "base"

    @abstractmethod
    def extract(self, text: str, allowed_labels: dict, context: dict | None = None) -> dict:
        raise NotImplementedError

    def extract_record(
        self,
        text: str,
        allowed_labels: dict,
        context: dict | None = None,
    ) -> dict:
        prediction = self.extract(text, allowed_labels, context=context)
        return {
            "raw_responses": {},
            "parsed_output": prediction,
            "validated_output": prediction,
            "json_validity": {"all_json_valid": True},
            "validation": {
                "has_invalid_labels": False,
                "invalid_labels": {"type": [], "queue": [], "tags": []},
                "tags_outside_candidates": [],
            },
        }

    def validate_prediction(self, prediction: dict) -> bool:
        required_keys = {"type", "queue", "tags"}
        if not required_keys.issubset(prediction.keys()):
            missing = required_keys - prediction.keys()
            raise ValueError(f"Prediction missing keys: {missing}")

        for field in ("type", "queue"):
            entry = prediction[field]
            if not isinstance(entry, dict) or "label" not in entry or "evidence" not in entry:
                raise ValueError(f"'{field}' must contain 'label' and 'evidence'")

        if not isinstance(prediction["tags"], list):
            raise ValueError("'tags' must be a list")

        for tag in prediction["tags"]:
            if not isinstance(tag, dict) or "label" not in tag or "evidence" not in tag:
                raise ValueError("Each tag must contain 'label' and 'evidence'")

        return True

    def validate_against_labels(
        self,
        prediction: dict | None,
        allowed_labels: dict,
        candidate_tags: list[str] | None = None,
        top_k_tags: int | None = None,
    ) -> tuple[dict, dict]:
        cleaned, validation = validate_and_clean_prediction(
            prediction=prediction,
            allowed_labels=allowed_labels,
            candidate_tags=candidate_tags,
            top_k_tags=top_k_tags,
        )
        self.validate_prediction(cleaned)
        return cleaned, validation

