"""Mechanical tests for the annotation tool's processed SciREX loader."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.benchmark_data import load_scirex_examples, source_metadata


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


record = {
    "example_id": "doc__sent_000000_000001", "doc_id": "doc", "source_split": "dev",
    "length_bucket": "short", "sentence_count": 1, "text": "A method.",
    "sentences": [], "sections": [], "entities": [],
}
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "fixture.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    loaded = load_scirex_examples(path)
    check("fixture loads", loaded == [record])
    metadata = source_metadata(loaded[0])
    check("source identifiers retained", metadata["example_id"] == record["example_id"])
    check("gold is not copied into source metadata", "entities" not in metadata)
