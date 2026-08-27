"""
This file asks the LLM to find entities in text, and turns its reply into a
clean list of entities the rest of the app can use.

Four methods are offered — really just four ways of asking the same thing:
- **Zero-shot** vs **few-shot**: zero-shot just describes the task; few-shot
  also shows 2 worked examples first, usually improving accuracy.
- **Freeform** vs **structured**: the *shape* of the reply. Freeform asks
  for plain text lines (`Munich | location | 6 | 6`); structured asks for
  JSON (`{"entities": [...]}`), which is stricter and easier to parse.

The extraction pipeline supports arbitrary labels selected by the user rather
than assuming a fixed, single-label task.

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

from utils.llm_client import call_llm_full, resolve_model_string
from utils.model_registry import ModelChoice
from utils.tokenizer import Sentence, Token, indexed_tokens_str, split_sentences, token_span_to_char_span, tokenize
from utils.uncertainty import aggregate_votes, confidence_from_logprobs

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
    "scirex_few_shot_structured": {
        "label": "SciREX few-shot — structured JSON",
        "description": "SciREX-specific label definitions and training-split examples for scientific NER.",
        "few_shot": True,
        "structured": True,
        "eval_csv_prefix": "scirex_few_shot_structured",
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

# These examples come from SciREX's training split, never dev/test. They use
# the dataset's own span conventions and jointly cover all four labels.
_RAW_SCIREX_EXAMPLES = [
    {
        "sentence": (
            "After training on Microsoft COCO, we compare our model with several baseline "
            "generative models on image generation and retrieval tasks."
        ),
        "entities": [
            ("Microsoft COCO", "Material"),
            ("baseline generative models", "Method"),
            ("image generation", "Task"),
            ("retrieval tasks", "Task"),
        ],
    },
    {
        "sentence": (
            "In the least-squares regression setting, typical in SR, the mean squared error "
            "averaged over the training set is minimized."
        ),
        "entities": [
            ("least-squares regression setting", "Method"),
            ("SR", "Task"),
            ("mean squared error", "Metric"),
        ],
    },
]


def _build_examples(raw_examples: list[dict]) -> list[dict]:
    built = []
    for raw in raw_examples:
        tokens = tokenize(raw["sentence"])
        entities = []
        for phrase, entity_type in raw["entities"]:
            start, end = _find_token_span(tokens, phrase)
            entities.append({"text": phrase, "type": entity_type, "start": start, "end": end})
        built.append({"sentence": raw["sentence"], "tokens": tokens, "gold_entities": entities})
    return built


_SCIREX_EXAMPLES = _build_examples(_RAW_SCIREX_EXAMPLES)


def _format_examples_freeform() -> str:
    """Format the examples the same way the freeform prompts expect the model's own output."""
    parts = []
    for ex in _FEW_SHOT_EXAMPLES:
        indexed = indexed_tokens_str(ex["tokens"])
        lines = "\n".join(f"{e['text']} | {e['type']} | {e['start']} | {e['end']}" for e in ex["gold_entities"])
        parts.append(f"Sentence: {ex['sentence']}\n\nToken indices:\n{indexed}\n\nOutput:\n{lines}")
    return "\n\n---\n\n".join(parts)


def _format_examples_structured(examples: list[dict] | None = None) -> str:
    """Same as above, but formatted as JSON for the structured prompts."""
    parts = []
    for ex in examples or _FEW_SHOT_EXAMPLES:
        indexed = indexed_tokens_str(ex["tokens"])
        output = json.dumps({"entities": ex["gold_entities"]}, ensure_ascii=False)
        parts.append(f"Sentence: {ex['sentence']}\n\nToken indices:\n{indexed}\n\nOutput: {output}")
    return "\n\n---\n\n".join(parts)


_EXAMPLES_FREEFORM = _format_examples_freeform()
_EXAMPLES_STRUCTURED = _format_examples_structured()
_SCIREX_EXAMPLES_STRUCTURED = _format_examples_structured(_SCIREX_EXAMPLES)


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
    token_logprobs: list = field(default_factory=list)  # per-token confidence, when the endpoint sends it
    model_id: str = ""  # which (provider, model) produced this
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    usage_reported: bool = False


def _build_prompt(
    method_id: str, labels: list[str], sentence_text: str, indexed_tokens: str,
    prompt_template: str | None = None,
) -> str:
    """Fill in one method's prompt template with the sentence, labels, and (for few-shot) examples."""
    template = prompt_template or _load_template(method_id)
    kwargs = {
        "labels": _format_labels(labels),
        "sentence": sentence_text,
        "indexed_tokens": indexed_tokens,
    }
    if METHODS[method_id]["few_shot"]:
        if method_id == "scirex_few_shot_structured":
            kwargs["examples"] = _SCIREX_EXAMPLES_STRUCTURED
        else:
            kwargs["examples"] = _EXAMPLES_STRUCTURED if METHODS[method_id]["structured"] else _EXAMPLES_FREEFORM
    return template.format(**kwargs)  # replaces each {placeholder} with its matching value


def extract_sentence(
    method_id: str,
    labels: list[str],
    sentence: Sentence,
    llm_params: dict | None = None,
    choice: ModelChoice | None = None,
    want_logprobs: bool = False,
    prompt_template: str | None = None,
) -> SentenceResult:
    """Run one method on a single sentence, returning the entities found (with
    character positions) plus anything rejected as invalid.

    `choice` picks which (provider, model) to call; `want_logprobs` asks the
    endpoint for token confidences so the caller can score each entity.
    """
    llm_params = dict(llm_params or {})
    llm_params.pop("model", None)  # the model now comes from `choice`
    indexed_tokens = indexed_tokens_str(sentence.tokens)
    prompt = _build_prompt(method_id, labels, sentence.text, indexed_tokens, prompt_template)
    response = call_llm_full(prompt, choice=choice, want_logprobs=want_logprobs, **llm_params)
    raw_response = response.text

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
    result.token_logprobs = response.token_logprobs
    result.model_id = response.model_id
    result.input_tokens = response.input_tokens
    result.output_tokens = response.output_tokens
    result.total_tokens = response.total_tokens
    result.usage_reported = response.usage_reported
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


@dataclass
class Pass:
    """One trip through the document: which model, at which temperature.

    Every uncertainty estimator is expressed as a list of passes, which keeps
    run_annotation() simple — it does not need to know *why* it is running
    three times, only that it is.
    """

    run_id: str                       # shown in the UI: "run 2", or a model id
    choice: ModelChoice | None = None
    temperature: float | None = None  # None means "use whatever llm_params says"
    seed: int | None = None


def build_passes(
    estimator: str,
    base_choice: ModelChoice | None,
    n_samples: int = 3,
    sample_temperature: float = 0.7,
    compare_choices: list[ModelChoice] | None = None,
) -> tuple[list[Pass], bool]:
    """Turn an estimator name into the list of passes to run.

    Returns (passes, want_logprobs).
    """
    if estimator == "self_consistency":
        # Temperature has to be above zero or every sample is identical and
        # the vote count carries no information.
        return (
            [
                Pass(run_id=f"run {i + 1}", choice=base_choice, temperature=sample_temperature, seed=1000 + i)
                for i in range(max(2, n_samples))
            ],
            False,
        )

    if estimator == "model_agreement":
        choices = compare_choices or []
        if len(choices) < 2:
            raise ValueError("Model agreement needs at least two models selected.")
        return ([Pass(run_id=c.id, choice=c) for c in choices], False)

    if estimator == "logprob":
        return ([Pass(run_id="run 1", choice=base_choice)], True)

    # "none"
    return ([Pass(run_id="run 1", choice=base_choice)], False)


def run_annotation(
    text: str,
    labels: list[str],
    method_id: str,
    max_sentences: int | None = 12,
    llm_params: dict | None = None,
    progress_callback=None,
    estimator: str = "none",
    passes: list[Pass] | None = None,
    want_logprobs: bool = False,
    prompt_template: str | None = None,
) -> dict:
    """Run the chosen pre-labeling method over a whole document, one sentence
    at a time (keeps prompts short and makes it easy to cap LLM calls on
    long documents via `max_sentences`).

    Args:
        text: the full document text.
        labels: entity labels to look for, e.g. ["location", "person"].
        method_id: which of the four METHODS to use.
        max_sentences: stop after this many sentences, or process all when None.
        llm_params: extra settings (temperature, ...) passed to the LLM call.
        progress_callback: optional function called as (index, total, sentence_text)
            after each sentence starts, for a live progress display.
        estimator: which uncertainty estimator to use — see utils/uncertainty.py.
        passes: the runs to make, from build_passes(). Defaults to a single run.
        want_logprobs: ask the endpoint for token confidences.

    Returns a dict with the combined entity list (document-level character
    positions), each entity carrying a `confidence` between 0 and 1 where one
    could be computed, plus a few extra details for debugging.
    """
    sentences = split_sentences(text, max_sentences=max_sentences)
    passes = passes or [Pass(run_id="run 1")]

    entities: list[dict] = []
    sentence_results: list[SentenceResult] = []
    # Per pass, the entities it found across the whole document. Used by the
    # comparison page so it can show the two models side by side.
    per_pass_entities: dict[str, list[dict]] = {p.run_id: [] for p in passes}
    logprobs_seen = False

    total_steps = len(sentences) * len(passes)
    step = 0

    for i, sentence in enumerate(sentences):
        if progress_callback:
            progress_callback(i, len(sentences), sentence.text)

        runs_for_sentence: list[tuple[str, list[dict]]] = []
        first_result: SentenceResult | None = None

        for p in passes:
            params = dict(llm_params or {})
            if p.temperature is not None:
                params["temperature"] = p.temperature
            if p.seed is not None:
                params["seed"] = p.seed

            result = extract_sentence(
                method_id, labels, sentence,
                llm_params=params, choice=p.choice, want_logprobs=want_logprobs,
                prompt_template=prompt_template,
            )
            step += 1
            if first_result is None:
                first_result = result
            sentence_results.append(result)
            runs_for_sentence.append((p.run_id, result.entities))
            per_pass_entities[p.run_id].extend(result.entities)

            if result.token_logprobs:
                logprobs_seen = True

        entities.extend(
            _score_sentence(runs_for_sentence, first_result, estimator, want_logprobs)
        )

    return {
        "entities": entities,
        "sentence_results": sentence_results,
        "per_pass_entities": per_pass_entities,
        "pass_ids": [p.run_id for p in passes],
        "estimator": estimator,
        "n_passes": len(passes),
        "n_llm_calls": total_steps,
        "input_tokens": sum(item.input_tokens for item in sentence_results),
        "output_tokens": sum(item.output_tokens for item in sentence_results),
        "total_tokens": sum(item.total_tokens for item in sentence_results),
        "usage_reported": all(item.usage_reported for item in sentence_results) if sentence_results else False,
        # False here after a logprob run means the endpoint ignored the
        # request, which the UI reports so the number is not silently missing.
        "logprobs_available": logprobs_seen,
        "n_sentences": len(sentences),
        # Recompute the *total* sentence count with no limit, so the page can
        # show "processed 6 of 42" when max_sentences cut the run short.
        "n_sentences_total": len(split_sentences(text)) if max_sentences else len(sentences),
        "processed_char_end": max(
            (sentence.doc_start + len(sentence.text) for sentence in sentences), default=0
        ),
    }


def _score_sentence(
    runs: list[tuple[str, list[dict]]],
    first_result: SentenceResult | None,
    estimator: str,
    want_logprobs: bool,
) -> list[dict]:
    """Merge one sentence's runs into a scored entity list.

    Every returned entity gets three extra keys:
      confidence  - 0..1, or None when we could not score it
      conf_source - which estimator produced the number, for the audit trail
      voters      - which runs/models found this span
    """
    if not runs:
        return []

    # --- single run: score from token logprobs if we have them -------------
    if len(runs) == 1:
        run_id, found = runs[0]
        scored = []
        for entity in found:
            confidence = None
            source = "none"
            if want_logprobs and first_result is not None and first_result.token_logprobs:
                confidence = confidence_from_logprobs(
                    entity, first_result.token_logprobs, first_result.raw_response
                )
                if confidence is not None:
                    source = "logprob"
            scored.append({**entity, "confidence": confidence, "conf_source": source, "voters": [run_id]})
        return scored

    # --- several runs: score by how many of them found each span ----------
    source = "model_agreement" if estimator == "model_agreement" else "self_consistency"
    return [
        {
            **vote.entity,
            "confidence": vote.confidence,
            "conf_source": source,
            "voters": vote.voters,
        }
        for vote in aggregate_votes(runs)
    ]
