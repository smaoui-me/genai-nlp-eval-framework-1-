"""
Splits text into sentences, and each sentence into "tokens" (words and
punctuation marks), remembering exactly where each token sits in the
original text.

Why: we ask the LLM to point at entities using *token numbers* (e.g. "word
4 to word 6"), since that's more reliable for LLMs than character
positions. But to highlight text and save results, we need character
positions. This file converts between the two.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches one "word" (letters/numbers, allowing an internal - or ') or one
# punctuation character at a time.
_TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]")

# A rough sentence-boundary pattern: split after ./!/? followed by a
# capital letter, digit, or quote, or on a line break.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|\n+")


# @dataclass auto-generates the boilerplate (like __init__) for a class
# that just stores a few named values.
@dataclass
class Token:
    text: str
    start: int
    end: int  # exclusive: the character AT this index is not part of the token


@dataclass
class Sentence:
    text: str
    doc_start: int  # this sentence's starting position in the whole document
    tokens: list[Token]


def tokenize(text: str, doc_offset: int = 0) -> list[Token]:
    """Split `text` into Tokens with character positions.
    `doc_offset` shifts those positions, for tokenizing a piece of a bigger document."""
    tokens = []
    for match in _TOKEN_RE.finditer(text):
        tokens.append(Token(text=match.group(), start=doc_offset + match.start(), end=doc_offset + match.end()))
    return tokens


def split_sentences(text: str, max_sentences: int | None = None) -> list[Sentence]:
    """Split a document into Sentences, each with its own tokens. Stops early
    if `max_sentences` is given, to limit how many LLM calls get made."""
    sentences: list[Sentence] = []
    cursor = 0
    for chunk in _SENTENCE_RE.split(text):
        if not chunk:
            continue
        start = text.index(chunk, cursor)
        cursor = start + len(chunk)

        stripped = chunk.strip()
        if not stripped:
            continue
        strip_offset = chunk.index(stripped)  # how much whitespace we trimmed off the front
        sentence_start = start + strip_offset

        tokens = tokenize(stripped, doc_offset=sentence_start)
        if not tokens:
            continue
        sentences.append(Sentence(text=stripped, doc_start=sentence_start, tokens=tokens))

        if max_sentences is not None and len(sentences) >= max_sentences:
            break
    return sentences


def indexed_tokens_str(tokens: list[Token]) -> str:
    """Format as "0: word\\n1: word\\n..." — what we show the LLM so it can refer to tokens by number."""
    return "\n".join(f"{i}: {tok.text}" for i, tok in enumerate(tokens))


def token_span_to_char_span(tokens: list[Token], start_tok: int, end_tok: int) -> tuple[int, int] | None:
    """Convert a token range (e.g. "token 4 to 6") into a character range.
    Returns None if the range is invalid (e.g. out of bounds)."""
    if not (0 <= start_tok <= end_tok < len(tokens)):
        return None
    return tokens[start_tok].start, tokens[end_tok].end
