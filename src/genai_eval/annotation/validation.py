"""Structural and offset validation for normalized SciREX artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    return records


def _validate_span(text: str, item: dict, label: str) -> None:
    start, end = item["start_char"], item["end_char"]
    if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(text)):
        raise ValueError(f"Invalid {label} character span [{start}, {end})")
    if "text" in item and text[start:end] != item["text"]:
        raise ValueError(f"{label} text mismatch: {text[start:end]!r} != {item['text']!r}")


def validate_documents(documents: list[dict]) -> None:
    ids = [doc.get("doc_id") for doc in documents]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Normalized doc_id values must be present and unique")
    for doc in documents:
        text, token_count = doc["text"], doc["token_count"]
        if token_count != len(doc["tokens"]):
            raise ValueError(f"Token count mismatch in {doc['doc_id']}")
        for index, token in enumerate(doc["tokens"]):
            if token["token_index"] != index:
                raise ValueError(f"Token order mismatch in {doc['doc_id']}")
            _validate_span(text, token, "token")
        for collection, name in ((doc["sentences"], "sentence"), (doc["sections"], "section"), (doc["entities"], "entity")):
            previous = -1
            for item in collection:
                start, end = item["token_start"], item["token_end_exclusive"]
                if not (0 <= start < end <= token_count):
                    raise ValueError(f"Invalid {name} token span in {doc['doc_id']}")
                if name in {"sentence", "section"} and start < previous:
                    raise ValueError(f"Unsorted {name} spans in {doc['doc_id']}")
                _validate_span(text, item, name)
                expected_start = doc["tokens"][start]["start_char"]
                expected_end = doc["tokens"][end - 1]["end_char"]
                if item["start_char"] != expected_start or item["end_char"] != expected_end:
                    raise ValueError(f"{name} token/character boundary mismatch in {doc['doc_id']}")
                previous = start


def validate_examples(examples: list[dict], documents: list[dict], config: dict) -> None:
    document_map = {doc["doc_id"]: doc for doc in documents}
    ids = [example.get("example_id") for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate benchmark example_id")
    windows = set()
    for example in examples:
        doc = document_map.get(example["doc_id"])
        if doc is None:
            raise ValueError(f"Unknown benchmark doc_id: {example['doc_id']}")
        if example["source_split"] != doc["source_split"]:
            raise ValueError(f"Split mismatch for {example['example_id']}")
        key = (example["doc_id"], example["source_sentence_start"], example["source_sentence_end_exclusive"])
        if key in windows:
            raise ValueError(f"Duplicate source window: {key}")
        windows.add(key)
        sentences = example["sentences"]
        if example["sentence_count"] != len(sentences):
            raise ValueError(f"Sentence count mismatch for {example['example_id']}")
        source_indices = [sentence["source_sentence_index"] for sentence in sentences]
        expected = list(range(example["source_sentence_start"], example["source_sentence_end_exclusive"]))
        if source_indices != expected:
            raise ValueError(f"Non-contiguous sentences in {example['example_id']}")
        source_sentences = doc["sentences"][example["source_sentence_start"]:example["source_sentence_end_exclusive"]]
        expected_text = doc["text"][source_sentences[0]["start_char"]:source_sentences[-1]["end_char"]]
        if example["text"] != expected_text:
            raise ValueError(f"Window text does not match source document in {example['example_id']}")
        spec = config["buckets"][example["length_bucket"]]
        count = example["sentence_count"]
        if count < spec["min_sentences"] or (spec.get("max_sentences") is not None and count > spec["max_sentences"]):
            raise ValueError(f"Bucket violation in {example['example_id']}")
        for sentence in sentences:
            _validate_span(example["text"], sentence, "example sentence")
        for entity in example["entities"]:
            _validate_span(example["text"], entity, "example entity")
            if not (
                example["source_token_start"] <= entity["source_token_start"]
                < entity["source_token_end_exclusive"] <= example["source_token_end_exclusive"]
            ):
                raise ValueError(f"Partially clipped entity in {example['example_id']}")
        for section in example["sections"]:
            _validate_span(example["text"], section, "example section")

    if len(examples) != int(config["target_examples"]):
        raise ValueError(f"Expected {config['target_examples']} examples, got {len(examples)}")
    counts = Counter(example["length_bucket"] for example in examples)
    expected_counts = {name: int(config["buckets"][name]["target"]) for name in config["buckets"]}
    if dict(counts) != expected_counts:
        raise ValueError(f"Bucket counts {dict(counts)} != {expected_counts}")

    maximum_overlap = float(config.get("sampling", {}).get("maximum_overlap_ratio", 0.5))
    by_document = {}
    for example in examples:
        by_document.setdefault(example["doc_id"], []).append(
            (example["source_sentence_start"], example["source_sentence_end_exclusive"])
        )
    for doc_id, source_windows in by_document.items():
        for index, first in enumerate(source_windows):
            for second in source_windows[index + 1:]:
                intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]))
                ratio = intersection / min(first[1] - first[0], second[1] - second[0])
                if ratio > maximum_overlap:
                    raise ValueError(
                        f"Window overlap {ratio:.3f} exceeds {maximum_overlap:.3f} for {doc_id}"
                    )


def validate_processed_dataset(processed_dir: str | Path, config: dict, fixture_dir: str | Path | None = None) -> dict:
    root = Path(processed_dir)
    documents = read_jsonl(root / "documents.jsonl")
    examples = read_jsonl(root / "examples_1000.jsonl")
    validate_documents(documents)
    validate_examples(examples, documents, config)
    combined = {json.dumps(item, sort_keys=True) for item in examples}
    split_union = set()
    split_record_count = 0
    for split in ("train", "dev", "test"):
        records = read_jsonl(root / f"{split}.jsonl")
        if any(record["source_split"] != split for record in records):
            raise ValueError(f"Wrong source split in {split}.jsonl")
        split_record_count += len(records)
        split_union.update(json.dumps(item, sort_keys=True) for item in records)
    if split_union != combined or split_record_count != len(examples):
        raise ValueError("Split file union does not equal examples_1000.jsonl")

    if fixture_dir:
        fixture_root = Path(fixture_dir)
        fixture = read_jsonl(fixture_root / "annotation_tool_smoke.jsonl")
        expected = json.loads((fixture_root / "annotation_tool_smoke_expected.json").read_text(encoding="utf-8"))
        if len(fixture) != expected["expected_example_count"]:
            raise ValueError("Smoke fixture count mismatch")
        counts = Counter(item["length_bucket"] for item in fixture)
        if dict(counts) != expected["expected_buckets"]:
            raise ValueError("Smoke fixture bucket mismatch")
        for item in fixture:
            missing = set(expected["required_fields"]) - set(item)
            if missing:
                raise ValueError(f"Smoke fixture missing fields: {sorted(missing)}")
    return {"documents": len(documents), "examples": len(examples), "unique_doc_ids": len({x['doc_id'] for x in examples})}
