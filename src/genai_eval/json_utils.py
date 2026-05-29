"""
JSON parsing helpers for LLM responses.
"""

import json
import re


def strip_json_fences(text: str) -> str:
    """Remove common markdown fences around JSON."""
    if text is None:
        return ""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_json_response(text: str) -> tuple[dict | list | None, bool, str | None]:
    """Parse JSON from an LLM response without raising.

    Returns:
        parsed object or None, validity flag, and error string if invalid.
    """
    cleaned = strip_json_fences(text)
    try:
        return json.loads(cleaned), True, None
    except json.JSONDecodeError as exc:
        return None, False, str(exc)
