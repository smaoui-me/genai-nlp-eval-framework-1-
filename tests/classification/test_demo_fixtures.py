import csv
import json
from pathlib import Path

from genai_eval.extraction.labels import COARSE_LABELS
from genai_eval.extraction.utils import (
    build_location_spans,
    parse_int_array,
    parse_token_array,
)


ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "sample_data" / "classification" / "embedding_demo"


def read_csv(name: str) -> list[dict]:
    with (DEMO_DIR / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_embedding_demo_splits_are_disjoint_and_schema_compatible():
    named_splits = {
        name: read_csv(name)
        for name in (
            "reference_tickets.csv",
            "development_tickets.csv",
            "held_out_tickets.csv",
            "unlabeled_tickets.csv",
        )
    }
    assert {name: len(rows) for name, rows in named_splits.items()} == {
        "reference_tickets.csv": 20,
        "development_tickets.csv": 8,
        "held_out_tickets.csv": 12,
        "unlabeled_tickets.csv": 4,
    }

    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    for rows in named_splits.values():
        ids = {row["ticket_id"] for row in rows}
        texts = {" ".join(row["text"].lower().split()) for row in rows}
        assert len(ids) == len(rows)
        assert len(texts) == len(rows)
        assert seen_ids.isdisjoint(ids)
        assert seen_texts.isdisjoint(texts)
        seen_ids.update(ids)
        seen_texts.update(texts)


def test_labeled_demo_splits_use_reference_schema_and_valid_json_tags():
    reference = read_csv("reference_tickets.csv")
    allowed_types = {row["gold_type"] for row in reference}
    allowed_queues = {row["gold_queue"] for row in reference}
    allowed_tags = {tag for row in reference for tag in json.loads(row["gold_tags"])}

    for name in ("reference_tickets.csv", "development_tickets.csv", "held_out_tickets.csv"):
        for row in read_csv(name):
            tags = json.loads(row["gold_tags"])
            assert row["gold_type"] in allowed_types
            assert row["gold_queue"] in allowed_queues
            assert isinstance(tags, list)
            assert set(tags).issubset(allowed_tags)


def test_public_few_shot_examples_contain_valid_location_spans():
    path = ROOT / "sample_data" / "extraction" / "fewnerd_few_shot_examples.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    location_id = next(key for key, value in COARSE_LABELS.items() if value == "location")

    assert len(rows) == 3
    for row in rows:
        tokens = parse_token_array(row["tokens"])
        tags = parse_int_array(row["ner_tags"])
        assert len(tokens) == len(tags)
        assert build_location_spans(tokens, tags, location_id)
