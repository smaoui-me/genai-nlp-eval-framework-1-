"""Raw SciREX discovery, parsing, and full-document normalization."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .detokenization import detokenize_with_offsets
from .schemas import REQUIRED_RAW_FIELDS, SCHEMA_VERSION, token_span_to_chars


SPLITS = ("train", "dev", "test")


def find_raw_split_files(raw_dir: str | Path) -> dict[str, Path]:
    root = Path(raw_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"SciREX raw directory does not exist: {root}")
    found: dict[str, Path] = {}
    for split in SPLITS:
        matches = sorted(root.rglob(f"{split}.jsonl"))
        if not matches:
            raise FileNotFoundError(f"Missing required SciREX split {split}.jsonl below {root}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous {split}.jsonl copies: {matches}")
        found[split] = matches[0]
    return found


def read_raw_records(path: Path, split: str) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            missing = REQUIRED_RAW_FIELDS - set(record)
            if missing:
                raise ValueError(f"Missing fields {sorted(missing)} in {path}:{line_number}")
            record["_source_split"] = split
            records.append(record)
    return records


def _normalise_span_records(
    spans: list, tokens: list[dict], text: str, kind: str, doc_id: str,
) -> list[dict]:
    output = []
    previous_start = -1
    for index, span in enumerate(spans):
        if not isinstance(span, list) or len(span) < 2:
            raise ValueError(f"Invalid {kind} span in {doc_id}: {span!r}")
        start, end = span[0], span[1]
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"Non-integer {kind} span in {doc_id}: {span!r}")
        if start < previous_start:
            raise ValueError(f"Unsorted {kind} spans in {doc_id}")
        start_char, end_char = token_span_to_chars(tokens, start, end)
        item = {
            f"{kind}_index": index,
            "token_start": start,
            "token_end_exclusive": end,
            "start_char": start_char,
            "end_char": end_char,
        }
        if kind == "sentence":
            item["text"] = text[start_char:end_char]
        output.append(item)
        previous_start = start
    return output


def normalize_record(raw: dict, split: str) -> dict:
    doc_id = str(raw["doc_id"]).strip()
    if not doc_id:
        raise ValueError("SciREX record has an empty doc_id")
    words = raw["words"]
    if not isinstance(words, list):
        raise ValueError(f"words must be a list in {doc_id}")
    text, token_offsets = detokenize_with_offsets(words)
    tokens = [offset.to_dict() for offset in token_offsets]
    sentences = _normalise_span_records(raw["sentences"], tokens, text, "sentence", doc_id)
    sections = _normalise_span_records(raw["sections"], tokens, text, "section", doc_id)

    entities = []
    for index, span in enumerate(raw["ner"]):
        if not isinstance(span, list) or len(span) != 3:
            raise ValueError(f"Invalid NER span in {doc_id}: {span!r}")
        start, end, label = span
        start_char, end_char = token_span_to_chars(tokens, start, end)
        entities.append({
            "entity_id": f"{doc_id}__entity_{index:06d}",
            "label": str(label),
            "text": text[start_char:end_char],
            "token_start": start,
            "token_end_exclusive": end,
            "start_char": start_char,
            "end_char": end_char,
        })

    metadata = {
        key: raw[key] for key in ("n_ary_relations", "method_subrelations") if key in raw
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "source_dataset": "scirex",
        "source_split": split,
        "doc_id": doc_id,
        "text": text,
        "token_count": len(tokens),
        "sentence_count": len(sentences),
        "section_count": len(sections),
        "entity_count": len(entities),
        "tokens": tokens,
        "sentences": sentences,
        "sections": sections,
        "entities": entities,
        "metadata": metadata,
    }


def load_and_normalize(raw_dir: str | Path) -> tuple[list[dict], dict[str, Path]]:
    files = find_raw_split_files(raw_dir)
    documents = []
    seen = set()
    for split in SPLITS:
        for raw in read_raw_records(files[split], split):
            document = normalize_record(raw, split)
            if document["doc_id"] in seen:
                raise ValueError(f"Duplicate SciREX doc_id: {document['doc_id']}")
            seen.add(document["doc_id"])
            documents.append(document)
    documents.sort(key=lambda item: (item["source_split"], item["doc_id"]))
    return documents, files


def observed_labels(documents: list[dict]) -> Counter:
    return Counter(entity["label"] for doc in documents for entity in doc["entities"])
