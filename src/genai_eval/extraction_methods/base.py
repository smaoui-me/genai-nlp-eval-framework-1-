"""
base.py

Abstract base class for all ticket extraction methods.
Every method must implement the `extract` interface and return
a standardized prediction dictionary.
"""

from abc import ABC, abstractmethod


class ExtractionMethod(ABC):
    """Base class for label-grounded ticket extraction methods.

    All subclasses must set a `name` class attribute and implement `extract`.
    The returned dict must follow the standard prediction schema:

        {
            "type": {"label": "...", "evidence": "..."},
            "queue": {"label": "...", "evidence": "..."},
            "tags": [{"label": "...", "evidence": "..."}, ...]
        }
    """

    name: str = "base"

    @abstractmethod
    def extract(self, text: str, allowed_labels: dict) -> dict:
        """Run extraction on a single ticket text.

        Args:
            text: Combined subject + body of the ticket.
            allowed_labels: Dict with keys "types", "queues", "tags",
                each containing a list of valid label strings.

        Returns:
            Prediction dict with "type", "queue", and "tags" keys.
        """
        raise NotImplementedError

    def validate_prediction(self, prediction: dict) -> bool:
        """Check that a prediction dict matches the required schema.

        Args:
            prediction: Output of extract().

        Returns:
            True if valid, raises ValueError otherwise.
        """
        required_keys = {"type", "queue", "tags"}
        if not required_keys.issubset(prediction.keys()):
            missing = required_keys - prediction.keys()
            raise ValueError(f"Prediction missing keys: {missing}")

        for field in ("type", "queue"):
            entry = prediction[field]
            if not isinstance(entry, dict) or "label" not in entry or "evidence" not in entry:
                raise ValueError(
                    f"'{field}' must be a dict with 'label' and 'evidence' keys, got: {entry}"
                )

        if not isinstance(prediction["tags"], list):
            raise ValueError("'tags' must be a list")

        for tag in prediction["tags"]:
            if not isinstance(tag, dict) or "label" not in tag or "evidence" not in tag:
                raise ValueError(
                    f"Each tag must be a dict with 'label' and 'evidence' keys, got: {tag}"
                )

        return True
