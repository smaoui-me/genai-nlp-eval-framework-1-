"""Load processed SciREX examples without exposing raw release objects to UI code."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SCIREX_FIXTURE = APP_DIR.parent / "data" / "annotation" / "fixtures" / "scirex" / "annotation_tool_smoke.jsonl"
DEFAULT_SCIREX_BENCHMARK = APP_DIR.parent / "data" / "annotation" / "processed" / "scirex" / "examples_1000.jsonl"
REQUIRED_FIELDS = {
    "example_id", "doc_id", "source_split", "length_bucket", "text",
    "sentences", "sections", "entities",
}


@lru_cache(maxsize=8)
def load_scirex_examples(path: str | Path = DEFAULT_SCIREX_FIXTURE) -> list[dict]:
    source = Path(path)
    if not source.exists():
        return []
    examples = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            missing = REQUIRED_FIELDS - set(record)
            if missing:
                raise ValueError(f"SciREX fixture line {line_number} missing {sorted(missing)}")
            examples.append(record)
    return examples


def load_scirex_benchmark() -> list[dict]:
    """Load the complete prepared benchmark, falling back to the smoke fixture."""
    examples = load_scirex_examples(DEFAULT_SCIREX_BENCHMARK)
    return examples or load_scirex_examples(DEFAULT_SCIREX_FIXTURE)


def source_metadata(example: dict) -> dict:
    """Return only traceability fields safe to attach to an annotation export."""
    return {
        "source_dataset": "scirex",
        "example_id": example["example_id"],
        "doc_id": example["doc_id"],
        "source_split": example["source_split"],
        "length_bucket": example["length_bucket"],
        "sentence_count": example["sentence_count"],
    }


def find_scirex_example(example_id: str) -> dict | None:
    """Resolve an exported example without loading gold into the annotation page."""
    for path in (DEFAULT_SCIREX_FIXTURE, DEFAULT_SCIREX_BENCHMARK):
        for example in load_scirex_examples(path):
            if example["example_id"] == example_id:
                return example
    return None
