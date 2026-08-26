"""Small schema helpers shared by SciREX preprocessing modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass


SCHEMA_VERSION = "1.0"
REQUIRED_RAW_FIELDS = {"doc_id", "words", "sentences", "sections", "ner"}


@dataclass(frozen=True)
class TokenOffset:
    token_index: int
    text: str
    start_char: int
    end_char: int

    def to_dict(self) -> dict:
        return asdict(self)


def token_span_to_chars(tokens: list[dict], start: int, end_exclusive: int) -> tuple[int, int]:
    if not (0 <= start < end_exclusive <= len(tokens)):
        raise ValueError(
            f"Invalid exclusive token span [{start}, {end_exclusive}) for {len(tokens)} tokens"
        )
    return tokens[start]["start_char"], tokens[end_exclusive - 1]["end_char"]
