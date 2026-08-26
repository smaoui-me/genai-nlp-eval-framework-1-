"""Deterministic, dependency-free scientific-text detokenization."""

from __future__ import annotations

from .schemas import TokenOffset


NO_SPACE_BEFORE = {".", ",", ";", ":", "?", "!", "%", ")", "]", "}"}
NO_SPACE_AFTER = {"(", "[", "{"}
JOIN_BOTH = {"-", "–", "—", "/"}
CONTRACTION_SUFFIXES = {"'s", "'re", "'ve", "'ll", "'d", "'m", "n't"}


def _needs_separator(previous: str | None, token: str) -> bool:
    if previous is None:
        return False
    if token in NO_SPACE_BEFORE or token in JOIN_BOTH or previous in NO_SPACE_AFTER or previous in JOIN_BOTH:
        return False
    if token in CONTRACTION_SUFFIXES or token.startswith("'"):
        return False
    return True


def detokenize_with_offsets(tokens: list[str]) -> tuple[str, list[TokenOffset]]:
    """Reconstruct readable text while preserving every token byte-for-byte."""
    pieces: list[str] = []
    offsets: list[TokenOffset] = []
    cursor = 0
    previous: str | None = None
    for index, raw_token in enumerate(tokens):
        if not isinstance(raw_token, str) or not raw_token:
            raise ValueError(f"Token {index} must be a non-empty string")
        separator = " " if _needs_separator(previous, raw_token) else ""
        pieces.append(separator)
        cursor += len(separator)
        start = cursor
        pieces.append(raw_token)
        cursor += len(raw_token)
        offsets.append(TokenOffset(index, raw_token, start, cursor))
        previous = raw_token
    return "".join(pieces), offsets
