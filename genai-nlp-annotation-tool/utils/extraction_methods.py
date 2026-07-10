"""
This file asks the LLM to find entities in text, and turns its reply into a
clean list of entities the rest of the app can use.

Four methods are offered — really just four ways of asking the same thing:
- **Zero-shot** vs **few-shot**: zero-shot just describes the task; few-shot
  also shows 2 worked examples first, usually improving accuracy.
- **Freeform** vs **structured**: the *shape* of the reply. Freeform asks
  for plain text lines (`Munich | location | 6 | 6`); structured asks for
  JSON (`{"entities": [...]}`), which is stricter and easier to parse.

Adapted from the single-label version in
https://github.com/smaoui-me/genai-nlp-eval-framework-1-, generalized here
to support any labels the user picks.

Pipeline for one sentence: build a prompt -> call_llm() -> parse the reply
-> check it's valid -> convert the model's word-position answer into
character positions we can highlight.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from utils.llm_client import call_llm
from utils.tokenizer import Sentence, Token, indexed_tokens_str, split_sentences, token_span_to_char_span, tokenize

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Settings for each method, used to pick/build the right prompt below. The
# prompt *text* lives in separate prompts/*.txt files (one per method).
METHODS = {
    "zero_shot_freeform": {
        "label": "Zero-shot — freeform text",
        "description": "No examples. The model lists entities as plain `text | type | start | end` lines.",
        "few_shot": False,
        "structured": False,
        "eval_csv_prefix": "zero_shot_freeform",
    },
    "zero_shot_structured": {
        "label": "Zero-shot — structured JSON",
        "description": "No examples. The model returns a strict JSON object of entities.",
        "few_shot": False,
        "structured": True,
        "eval_csv_prefix": "zero_shot_structured",
    },
    "few_shot_freeform": {
        "label": "Few-shot — freeform text",
        "description": "Two worked examples included in the prompt. Output is plain text lines.",
        "few_shot": True,
        "structured": False,
        "eval_csv_prefix": "few_shot_freeform",
    },
    "few_shot_structured": {
        "label": "Few-shot — structured JSON",
        "description": "Two worked examples included in the prompt. Output is strict JSON.",
        "few_shot": True,
        "structured": True,
        "eval_csv_prefix": "few_shot_structured",
    },
}


# @lru_cache remembers a function's result and reuses it next time it's
# called with the same argument, instead of redoing the work. Here it means
# each prompt file is only read from disk once.
@lru_cache(maxsize=None)
def _load_template(method_id: str) -> str:
    """Read one method's prompt template file, placeholders like {sentence} still unfilled."""
    return (PROMPTS_DIR / f"{method_id}.txt").read_text(encoding="utf-8")


def _format_labels(labels: list[str]) -> str:
    """["location", "person"] -> "- location\\n- person", for the {labels} placeholder."""
    return "\n".join(f"- {label}" for label in labels)


# ---------------------------------------------------------------------------
# Few-shot examples — hand-written once below, reused in every few-shot prompt.
# ---------------------------------------------------------------------------

_RAW_FEW_SHOT_EXAMPLES = [
    {
        "sentence": (
            "Customer reported a delayed delivery in Munich after shipment from "
            "DHL Hub . Please contact John Miller before Friday and verify the "
            "destination address in Berlin ."
        ),
        "entities": [
            ("Munich", "location"),
            ("DHL Hub", "organization"),
            ("John Miller", "person"),
            ("Berlin", "location"),
        ],
    },
    {
        "sentence": (
            "Apple unveiled the new iPhone at their Cupertino headquarters during "
            "WWDC , with CEO Tim Cook presenting the keynote ."
        ),
        "entities": [
            ("Apple", "organization"),
            ("iPhone", "product"),
            ("Cupertino", "location"),
            ("WWDC", "event"),
            ("Tim Cook", "person"),
        ],
    },
]


def _find_token_span(tokens: list[Token], phrase: str) -> tuple[int, int]:
    """Find (start_token, end_token) for `phrase` inside `tokens`, by sliding
    a window across the sentence and checking for a match."""
    phrase_tokens = [tok.text for tok in tokenize(phrase)]
    n = len(phrase_tokens)
    for i in range(len(tokens) - n + 1):
        if [tok.text for tok in tokens[i : i + n]] == phrase_tokens:
            return i, i + n - 1
    raise ValueError(f"Example phrase not found in tokens: {phrase!r}")  # a typo in our own example data


def _build_few_shot_examples() -> list[dict]:
    """Turn the raw examples above into fully-worked ones (tokenized, with
    each entity's token position filled in)."""
    built = []
    for raw in _RAW_FEW_SHOT_EXAMPLES:
        tokens = tokenize(raw["sentence"])
        gold_entities = []
        for phrase, entity_type in raw["entities"]:
            start, end = _find_token_span(tokens, phrase)
            gold_entities.append({"text": phrase, "type": entity_type, "start": start, "end": end})
        built.append({"sentence": raw["sentence"], "tokens": tokens, "gold_entities": gold_entities})
    return built


_FEW_SHOT_EXAMPLES = _build_few_shot_examples()  # built once, at import time


def _format_examples_freeform() -> str:
    """Format the examples the same way the freeform prompts expect the model's own output."""
    parts = []
    for ex in _FEW_SHOT_EXAMPLES:
        indexed = indexed_tokens_str(ex["tokens"])
        lines = "\n".join(f"{e['text']} | {e['type']} | {e['start']} | {e['end']}" for e in ex["gold_entities"])
        parts.append(f"Sentence: {ex['sentence']}\n\nToken indices:\n{indexed}\n\nOutput:\n{lines}")
    return "\n\n---\n\n".join(parts)


def _format_examples_structured() -> str:
    """Same as above, but formatted as JSON for the structured prompts."""
    parts = []
    for ex in _FEW_SHOT_EXAMPLES:
        indexed = indexed_tokens_str(ex["tokens"])
        output = json.dumps({"entities": ex["gold_entities"]}, ensure_ascii=False)
        parts.append(f"Sentence: {ex['sentence']}\n\nToken indices:\n{indexed}\n\nOutput: {output}")
    return "\n\n---\n\n".join(parts)


_EXAMPLES_FREEFORM = _format_examples_freeform()
_EXAMPLES_STRUCTURED = _format_examples_structured()


# ---------------------------------------------------------------------------
# Response parsing — turning the LLM's raw text reply into Python data.
# ---------------------------------------------------------------------------


def _strip_json_fences(text: str) -> str:
    """LLMs often wrap JSON in ```json ... ``` code fences — remove those so
    the rest is plain JSON we can parse."""
    if text is None:
        return ""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_json_response(text: str) -> tuple[dict | list | None, bool, str | None]:
    """Try to parse `text` as JSON. Returns (value, was_it_valid, error) —
    a tuple instead of raising, since malformed JSON from an LLM is a
    normal, expected outcome here, not a bug."""
    cleaned = _strip_json_fences(text)
    try:
        return json.loads(cleaned), True, None
    except json.JSONDecodeError as exc:
        return None, False, str(exc)


def parse_freeform_entities(text: str) -> tuple[list[dict], list[dict]]:
    """Parse "text | type | start | end" lines. Returns (entities, malformed_lines)
    so problems are visible instead of silently dropped."""
    entities: list[dict] = []
    malformed: list[dict] = []
    cleaned = str(text or "").strip()
    if not cleaned or cleaned.upper() == "NONE":
        return entities, malformed

    for line in cleaned.splitlines():
        raw_line = line.strip().lstrip("-*").strip()
        if not raw_line:
            continue
        parts = [p.strip() for p in raw_line.split("|")]
        if len(parts) != 4:
            malformed.append({"line": raw_line, "reason": "wrong_field_count"})
            continue
        text_val, type_val, start_text, end_text = parts
        try:
            start, end = int(start_text), int(end_text)
        except ValueError:
            malformed.append({"line": raw_line, "reason": "non_integer_indices"})
            continue
        entities.append({"text": text_val, "type": type_val, "start": start, "end": end})

    return entities, malformed


def _normalize_type(raw_type: str, labels: list[str]) -> str | None:
    """Match the model's label text to one of our allowed labels, ignoring
    case/spacing. None if it doesn't match any — that entity gets dropped."""
    for label in labels:
        if label.strip().lower() == str(raw_type or "").strip().lower():
            return label
    return None


# ---------------------------------------------------------------------------
# Extraction — where everything above gets used together
# ---------------------------------------------------------------------------


# field(default_factory=list) gives each new SentenceResult its own empty
# list, instead of every instance accidentally sharing the same one.
@dataclass
class SentenceResult:
    sentence: Sentence
    raw_response: str
    entities: list[dict] = field(default_factory=list)  # found entities, with document-level character offsets
    invalid: list[dict] = field(default_factory=list)  # things we had to reject, and why
    json_valid: bool | None = None  # only meaningful for structured methods


def _build_prompt(method_id: str, labels: list[str], sentence_text: str, indexed_tokens: str) -> str:
    """Fill in one method's prompt template with the sentence, labels, and (for few-shot) examples."""
    template = _load_template(method_id)
    kwargs = {
        "labels": _format_labels(labels),
        "sentence": sentence_text,
        "indexed_tokens": indexed_tokens,
    }
    if METHODS[method_id]["few_shot"]:
        kwargs["examples"] = _EXAMPLES_STRUCTURED if METHODS[method_id]["structured"] else _EXAMPLES_FREEFORM
    return template.format(**kwargs)  # replaces each {placeholder} with its matching value


def extract_sentence(method_id: str, labels: list[str], sentence: Sentence, llm_params: dict | None = None) -> SentenceResult:
    """Run one method on a single sentence, returning the entities found (with
    character positions) plus anything rejected as invalid."""
    llm_params = llm_params or {}
    indexed_tokens = indexed_tokens_str(sentence.tokens)
    prompt = _build_prompt(method_id, labels, sentence.text, indexed_tokens)
    raw_response = call_llm(prompt, **llm_params)  # **llm_params spreads the dict into keyword arguments

    structured = METHODS[method_id]["structured"]
    malformed_lines: list[dict] = []
    if structured:
        parsed, json_valid, _ = parse_json_response(raw_response)
        raw_entities = parsed.get("entities", []) if isinstance(parsed, dict) else []
        raw_entities = [e for e in raw_entities if isinstance(e, dict)]  # ignore anything not a dict, defensively
    else:
        raw_entities, malformed_lines = parse_freeform_entities(raw_response)
        json_valid = None

    result = SentenceResult(sentence=sentence, raw_response=raw_response, json_valid=json_valid)
    n_tokens = len(sentence.tokens)

    for entity in raw_entities:
        entity_type = _normalize_type(entity.get("type", ""), labels)
        start_tok, end_tok = entity.get("start"), entity.get("end")
        text_val = str(entity.get("text", "")).strip()

        reasons = []  # collect every problem with this entity, not just the first one
        if entity_type is None:
            reasons.append("invalid_type")
        if not isinstance(start_tok, int) or not isinstance(end_tok, int):
            reasons.append("invalid_indices")
        elif not (0 <= start_tok <= end_tok < n_tokens):
            reasons.append("indices_out_of_range")

        if reasons:
            result.invalid.append({"entity": entity, "reason": reasons})
            continue

        char_span = token_span_to_char_span(sentence.tokens, start_tok, end_tok)
        if char_span is None:
            result.invalid.append({"entity": entity, "reason": ["indices_out_of_range"]})
            continue

        start_char, end_char = char_span
        result.entities.append(
            {
                # Prefer the model's own text; fall back to reading the
                # document at these positions if it left "text" blank.
                "text": text_val or sentence.text[start_char - sentence.doc_start : end_char - sentence.doc_start],
                "type": entity_type,
                "start": start_char,
                "end": end_char,
            }
        )

    if not structured and malformed_lines:
        for item in malformed_lines:
            result.invalid.append({"entity": item["line"], "reason": [item["reason"]]})

    return result


def run_annotation(
    text: str,
    labels: list[str],
    method_id: str,
    max_sentences: int = 12,
    llm_params: dict | None = None,
    progress_callback=None,
) -> dict:
    """Run the chosen pre-labeling method over a whole document, one sentence
    at a time (keeps prompts short and makes it easy to cap LLM calls on
    long documents via `max_sentences`).

    Args:
        text: the full document text.
        labels: entity labels to look for, e.g. ["location", "person"].
        method_id: which of the four METHODS to use.
        max_sentences: stop after this many sentences.
        llm_params: extra settings (temperature, model, ...) passed to call_llm().
        progress_callback: optional function called as (index, total, sentence_text)
            after each sentence starts, for a live progress display.

    Returns a dict with the combined entity list (document-level character
    positions) plus a few extra details for debugging.
    """
    sentences = split_sentences(text, max_sentences=max_sentences)
    entities: list[dict] = []
    sentence_results: list[SentenceResult] = []

    for i, sentence in enumerate(sentences):
        if progress_callback:
            progress_callback(i, len(sentences), sentence.text)
        result = extract_sentence(method_id, labels, sentence, llm_params=llm_params)
        sentence_results.append(result)
        entities.extend(result.entities)

    return {
        "entities": entities,
        "sentence_results": sentence_results,
        "n_sentences": len(sentences),
        # Recompute the *total* sentence count with no limit, so the page can
        # show "processed 6 of 42" when max_sentences cut the run short.
        "n_sentences_total": len(split_sentences(text)) if max_sentences else len(sentences),
    }
