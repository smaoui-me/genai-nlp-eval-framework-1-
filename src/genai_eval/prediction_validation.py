"""
Prediction validation and cleanup helpers.
"""


def _empty_field() -> dict:
    return {"label": "", "evidence": ""}


def _ensure_field_dict(value) -> dict:
    if not isinstance(value, dict):
        return _empty_field()
    return {
        "label": str(value.get("label", "")).strip(),
        "evidence": str(value.get("evidence", "")).strip(),
    }


def _ensure_tag_list(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        if not isinstance(item, dict):
            continue
        cleaned.append(
            {
                "label": str(item.get("label", "")).strip(),
                "evidence": str(item.get("evidence", "")).strip(),
            }
        )
    return cleaned


def validate_and_clean_prediction(
    prediction: dict | None,
    allowed_labels: dict,
    candidate_tags: list[str] | None = None,
    top_k_tags: int | None = None,
) -> tuple[dict, dict]:
    """Validate a parsed prediction and remove unsupported labels."""
    allowed_types = set(allowed_labels.get("types", []))
    allowed_queues = set(allowed_labels.get("queues", []))
    allowed_tags = set(allowed_labels.get("tags", []))
    candidate_tag_set = set(candidate_tags or [])

    prediction = prediction if isinstance(prediction, dict) else {}
    cleaned = {
        "type": _ensure_field_dict(prediction.get("type")),
        "queue": _ensure_field_dict(prediction.get("queue")),
        "tags": _ensure_tag_list(prediction.get("tags")),
    }

    invalid_labels = {"type": [], "queue": [], "tags": []}
    tags_outside_candidates = []

    if cleaned["type"]["label"] and cleaned["type"]["label"] not in allowed_types:
        invalid_labels["type"].append(cleaned["type"]["label"])
        cleaned["type"] = _empty_field()

    if cleaned["queue"]["label"] and cleaned["queue"]["label"] not in allowed_queues:
        invalid_labels["queue"].append(cleaned["queue"]["label"])
        cleaned["queue"] = _empty_field()

    validated_tags = []
    seen_labels = set()
    for tag in cleaned["tags"]:
        label = tag["label"]
        if not label or label in seen_labels:
            continue
        seen_labels.add(label)

        if label not in allowed_tags:
            invalid_labels["tags"].append(label)
            continue

        if candidate_tag_set and label not in candidate_tag_set:
            tags_outside_candidates.append(label)

        validated_tags.append(tag)

    if top_k_tags is not None and top_k_tags > 0:
        validated_tags = validated_tags[:top_k_tags]

    cleaned["tags"] = validated_tags

    validation = {
        "has_invalid_labels": any(invalid_labels.values()),
        "invalid_labels": invalid_labels,
        "tags_outside_candidates": tags_outside_candidates,
    }
    return cleaned, validation
