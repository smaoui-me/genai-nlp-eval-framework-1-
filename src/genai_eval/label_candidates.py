"""
Candidate tag selection helpers.
"""

import json
import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def normalize_token(token: str) -> str:
    return token.lower().strip()


def tokenize(text: str) -> list[str]:
    return [normalize_token(token) for token in TOKEN_PATTERN.findall(text or "")]


def split_label_words(label: str) -> list[str]:
    parts = re.split(r"[^A-Za-z0-9]+", label or "")
    return [normalize_token(part) for part in parts if part]


def is_noisy_tag(label: str) -> bool:
    """Filter obviously noisy tag labels from prompt candidate lists."""
    if not label or not label.strip():
        return True
    if label.count(",") >= 2:
        return True
    if len(label) > 60:
        return True
    return False


def parse_tag_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return []


def build_tag_frequency(gold_tag_values: list) -> dict[str, int]:
    counter = Counter()
    for value in gold_tag_values:
        counter.update(parse_tag_list(value))
    return dict(counter)


def score_tag_match(tag: str, text_tokens: set[str], text_lower: str) -> int:
    tag_words = split_label_words(tag)
    if not tag_words:
        return 0

    overlap = sum(1 for word in tag_words if word in text_tokens)
    phrase_bonus = 2 if tag.lower() in text_lower else 0

    if overlap == 0 and phrase_bonus == 0:
        return 0
    generic_bonus = 1 if len(tag_words) <= 2 else 0
    return overlap + phrase_bonus + generic_bonus


def select_candidate_tags(
    text: str,
    allowed_tags: list[str],
    tag_frequency: dict[str, int] | None = None,
    max_candidates: int = 40,
    fallback_top_k: int = 20,
) -> list[str]:
    """Select a manageable tag shortlist for one ticket.

    Strategy:
    - Keep tags whose words or full phrase appear in the ticket text.
    - Fill remaining slots with the most frequent tags from the dataset.
    - Fall back to the original allowed-tag order if still short.
    """
    text_lower = (text or "").lower()
    text_tokens = set(tokenize(text))
    tag_frequency = tag_frequency or {}

    scored = []
    for tag in allowed_tags:
        if is_noisy_tag(tag):
            continue
        score = score_tag_match(tag, text_tokens, text_lower)
        if score > 0:
            scored.append((tag, score, tag_frequency.get(tag, 0)))

    scored.sort(key=lambda item: (-item[1], -item[2], item[0].lower()))
    fallback_budget = min(max(fallback_top_k, 0), max_candidates)
    lexical_budget = max(max_candidates - fallback_budget, 0)
    selected = [tag for tag, _, _ in scored[:lexical_budget]]

    if len(selected) < max_candidates:
        frequent_tags = sorted(
            [tag for tag in allowed_tags if not is_noisy_tag(tag)],
            key=lambda tag: (-tag_frequency.get(tag, 0), tag.lower()),
        )
        for tag in frequent_tags[:fallback_budget]:
            if tag not in selected:
                selected.append(tag)
            if len(selected) >= max_candidates:
                break

    if len(selected) < max_candidates:
        for tag in allowed_tags:
            if is_noisy_tag(tag):
                continue
            if tag not in selected:
                selected.append(tag)
            if len(selected) >= max_candidates:
                break

    return selected[:max_candidates]
