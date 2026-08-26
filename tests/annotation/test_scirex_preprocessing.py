from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genai_eval.annotation.chunking import build_benchmark, bucket_for_sentence_count
from genai_eval.annotation.detokenization import detokenize_with_offsets
from genai_eval.annotation.scirex import find_raw_split_files, normalize_record, read_raw_records
from genai_eval.annotation.validation import validate_documents, validate_examples, validate_processed_dataset


CONFIG = {
    "seed": 42,
    "target_examples": 4,
    "buckets": {
        "short": {"min_sentences": 1, "max_sentences": 1, "target": 1},
        "medium": {"min_sentences": 2, "max_sentences": 2, "target": 1},
        "long": {"min_sentences": 3, "max_sentences": 3, "target": 1},
        "very_long": {"min_sentences": 4, "max_sentences": None, "target": 1},
    },
    "sampling": {"maximum_overlap_ratio": 0.5},
}


def raw_document(doc_id="doc", sentence_count=5, split="train"):
    words = []
    sentences = []
    ner = []
    for index in range(sentence_count):
        start = len(words)
        words.extend(["Method", str(index), "."])
        sentences.append([start, start + 3])
        ner.append([start, start + 1, "Method"])
    return {
        "doc_id": doc_id, "words": words, "sentences": sentences,
        "sections": [[0, len(words)]], "ner": ner, "_source_split": split,
    }


def test_detokenization_is_deterministic_and_preserves_tokens():
    tokens = ["A", "(", "method", ")", ",", "does", "n't", "fail", "."]
    first = detokenize_with_offsets(tokens)
    second = detokenize_with_offsets(tokens)
    assert first == second
    text, offsets = first
    assert text == "A (method), doesn't fail."
    assert all(text[item.start_char:item.end_char] == item.text for item in offsets)


def test_raw_record_parsing_and_required_fields(tmp_path):
    path = tmp_path / "train.jsonl"
    record = raw_document()
    path.write_text(json.dumps({k: v for k, v in record.items() if not k.startswith("_")}) + "\n", encoding="utf-8")
    parsed = read_raw_records(path, "train")
    assert parsed[0]["doc_id"] == "doc"
    path.write_text(json.dumps({"doc_id": "broken"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing fields"):
        read_raw_records(path, "train")


def test_recursive_split_discovery_rejects_ambiguity(tmp_path):
    for split in ("train", "dev", "test"):
        (tmp_path / f"{split}.jsonl").write_text("", encoding="utf-8")
    assert set(find_raw_split_files(tmp_path)) == {"train", "dev", "test"}
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "train.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Ambiguous"):
        find_raw_split_files(tmp_path)


def test_exclusive_end_and_character_conversion():
    doc = normalize_record(raw_document(sentence_count=2), "train")
    assert doc["sentences"][0]["token_end_exclusive"] == 3
    assert doc["sentences"][0]["text"] == "Method 0."
    section = doc["sections"][0]
    assert doc["text"][section["start_char"]:section["end_char"]] == doc["text"]
    entity = doc["entities"][0]
    assert entity["text"] == "Method"
    assert doc["text"][entity["start_char"]:entity["end_char"]] == entity["text"]
    validate_documents([doc])


def test_windowing_stays_in_document_and_removes_partial_entities():
    doc = normalize_record(raw_document(sentence_count=5), "dev")
    # Cross the first window boundary: this entity must be excluded.
    doc["entities"].append({
        "entity_id": "cross", "label": "Method", "text": doc["text"][0:12],
        "token_start": 0, "token_end_exclusive": 4, "start_char": 0, "end_char": 12,
    })
    documents = [doc] + [
        normalize_record(raw_document(f"doc-{index}", 5, "dev"), "dev") for index in range(1, 4)
    ]
    examples, _ = build_benchmark(documents, CONFIG)
    assert all(item["doc_id"] in {d["doc_id"] for d in documents} and item["source_split"] == "dev" for item in examples)
    for item in examples:
        assert all(not (entity["entity_id"] == "cross" and item["source_token_end_exclusive"] < 4)
                   for entity in item["entities"])
    validate_examples(examples, documents, CONFIG)


def test_bucket_classification():
    assert bucket_for_sentence_count(1, CONFIG["buckets"]) == "short"
    assert bucket_for_sentence_count(2, CONFIG["buckets"]) == "medium"
    assert bucket_for_sentence_count(3, CONFIG["buckets"]) == "long"
    assert bucket_for_sentence_count(4, CONFIG["buckets"]) == "very_long"


def test_sampling_is_deterministic_and_has_no_duplicate_windows():
    documents = [normalize_record(raw_document(f"doc-{i}", 6, "test"), "test") for i in range(4)]
    first, _ = build_benchmark(documents, CONFIG)
    second, _ = build_benchmark(documents, CONFIG)
    assert first == second
    keys = {(x["doc_id"], x["source_sentence_start"], x["source_sentence_end_exclusive"]) for x in first}
    assert len(keys) == len(first)
    assert all(item["source_split"] == "test" for item in first)


@pytest.mark.integration
def test_real_processed_fixture_when_present():
    processed = ROOT / "data" / "annotation" / "processed" / "scirex"
    fixture = ROOT / "data" / "annotation" / "fixtures" / "scirex"
    if not (processed / "examples_1000.jsonl").exists():
        pytest.skip("Processed SciREX artifacts not present")
    import yaml
    config = yaml.safe_load((ROOT / "configs" / "annotation" / "scirex_preprocessing.yaml").read_text())
    result = validate_processed_dataset(processed, config, fixture)
    assert result["examples"] == 1000
