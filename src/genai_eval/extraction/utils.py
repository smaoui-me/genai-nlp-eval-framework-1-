"""Utilities for FewNERD preprocessing and extraction span handling."""

from __future__ import annotations

import json
import re


def parse_token_array(value: str) -> list[str]:
    if not isinstance(value, str):
        return []
    tokens = []
    for single_quoted, double_quoted in re.findall(r"'([^']*)'|\"([^\"]*)\"", value):
        tokens.append(single_quoted if single_quoted else double_quoted)
    return tokens


def parse_int_array(value: str) -> list[int]:
    if not isinstance(value, str):
        return []
    return [int(item) for item in re.findall(r"-?\d+", value)]


def build_spans(tokens: list[str], tag_ids: list[int], label_map: dict[int, str]) -> list[dict]:
    spans: list[dict] = []
    start = None
    current_tag = 0

    for index, tag_id in enumerate(tag_ids + [0]):
        if tag_id != current_tag:
            if current_tag != 0 and start is not None:
                end = index - 1
                span_tokens = tokens[start : end + 1]
                spans.append(
                    {
                        "text": " ".join(span_tokens),
                        "type": label_map.get(current_tag, f"unknown-{current_tag}"),
                        "start": start,
                        "end": end,
                    }
                )
            if tag_id != 0:
                start = index
            else:
                start = None
            current_tag = tag_id

    return spans


def build_location_spans(tokens: list[str], coarse_tag_ids: list[int], location_tag_id: int) -> list[dict]:
    spans: list[dict] = []
    start = None

    for index, tag_id in enumerate(coarse_tag_ids + [0]):
        is_location = tag_id == location_tag_id
        if is_location and start is None:
            start = index
        elif not is_location and start is not None:
            end = index - 1
            spans.append(
                {
                    "text": " ".join(tokens[start : end + 1]),
                    "start": start,
                    "end": end,
                }
            )
            start = None

    return spans


def normalize_span_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return re.sub(r"\s+", " ", text.strip())


def sentence_from_tokens(tokens: list[str]) -> str:
    return " ".join(tokens)


def format_indexed_tokens(tokens: list[str]) -> str:
    return "\n".join(f"{index}: {token}" for index, token in enumerate(tokens))


def to_json_string(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def parse_freeform_entities(text: str) -> tuple[list[dict], list[dict]]:
    entities: list[dict] = []
    malformed_lines: list[dict] = []
    cleaned = str(text or "").strip()
    if not cleaned or cleaned.upper() == "NONE":
        return entities, malformed_lines

    for line in cleaned.splitlines():
        raw_line = line.strip()
        if not raw_line:
            continue
        parts = [part.strip() for part in raw_line.split("|")]
        if len(parts) != 3:
            malformed_lines.append({"line": raw_line, "reason": "wrong_field_count"})
            continue
        entity_text, start_text, end_text = parts
        try:
            start = int(start_text)
            end = int(end_text)
        except ValueError:
            malformed_lines.append({"line": raw_line, "reason": "non_integer_indices"})
            continue
        entities.append({"text": entity_text, "start": start, "end": end})

    return entities, malformed_lines
