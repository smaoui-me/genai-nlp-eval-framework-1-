"""Base class for extraction methods."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ExtractionMethod(ABC):
    name: str = "base"

    @abstractmethod
    def extract_record(self, tokens: list[str], allowed_entity_types: list[str]) -> dict:
        raise NotImplementedError

    @staticmethod
    def empty_prediction() -> dict:
        return {"entities": []}

    def validate_prediction(self, prediction: dict) -> bool:
        if not isinstance(prediction, dict):
            raise ValueError("Prediction must be a JSON object")
        entities = prediction.get("entities", [])
        if not isinstance(entities, list):
            raise ValueError("'entities' must be a list")
        for entity in entities:
            if not isinstance(entity, dict):
                raise ValueError("Each entity must be an object")
            required_keys = {"text", "type", "start", "end"}
            missing = required_keys - set(entity.keys())
            if missing:
                raise ValueError(f"Entity missing keys: {sorted(missing)}")
        return True

    def validate_against_labels(
        self,
        prediction: dict | None,
        allowed_entity_types: list[str],
        tokens: list[str],
    ) -> tuple[dict, dict]:
        cleaned = {"entities": []}
        invalid_entities = []
        has_invalid_labels = False
        allowed_set = set(allowed_entity_types)

        entities = prediction.get("entities", []) if isinstance(prediction, dict) else []
        if not isinstance(entities, list):
            entities = []

        for entity in entities:
            if not isinstance(entity, dict):
                invalid_entities.append({"entity": entity, "reason": "not_an_object"})
                has_invalid_labels = True
                continue

            text = str(entity.get("text", "")).strip()
            entity_type = str(entity.get("type", "")).strip()
            start = entity.get("start")
            end = entity.get("end")

            reasons = []
            if entity_type not in allowed_set:
                reasons.append("invalid_type")
            if not isinstance(start, int) or not isinstance(end, int):
                reasons.append("invalid_indices")
            elif start < 0 or end < start or end >= len(tokens):
                reasons.append("indices_out_of_range")
            if not text:
                reasons.append("empty_text")

            expected_text = " ".join(tokens[start : end + 1]) if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end < len(tokens) else ""
            if text and expected_text and text.strip() != expected_text.strip():
                reasons.append("text_span_mismatch")

            if reasons:
                invalid_entities.append({"entity": entity, "reason": reasons})
                has_invalid_labels = True
                continue

            cleaned["entities"].append(
                {
                    "text": text,
                    "type": entity_type,
                    "start": start,
                    "end": end,
                }
            )

        self.validate_prediction(cleaned)
        return cleaned, {"has_invalid_labels": has_invalid_labels, "invalid_entities": invalid_entities}
