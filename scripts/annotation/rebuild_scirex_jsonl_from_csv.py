"""Rebuild the runnable SciREX benchmark JSONL from the public linked CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild examples_1000.jsonl from examples, sentences, and entities CSV files."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("data/annotation/processed/scirex/csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/annotation/processed/scirex/examples_1000.jsonl"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def integer(row: dict, key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer in column {key!r}: {row.get(key)!r}") from exc


def grouped_sentences(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["example_id"]].append(
                {
                    "relative_sentence_index": integer(row, "relative_sentence_index"),
                    "source_sentence_index": integer(row, "source_sentence_index"),
                    "start_char": integer(row, "start_char"),
                    "end_char": integer(row, "end_char"),
                    "text": row["text"],
                }
            )
    for values in grouped.values():
        values.sort(key=lambda item: item["relative_sentence_index"])
    return grouped


def grouped_entities(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["example_id"]].append(
                {
                    "entity_id": row["entity_id"],
                    "label": row["label"],
                    "text": row["text"],
                    "start_char": integer(row, "start_char"),
                    "end_char": integer(row, "end_char"),
                    "source_token_start": integer(row, "source_token_start"),
                    "source_token_end_exclusive": integer(row, "source_token_end_exclusive"),
                }
            )
    return grouped


def build_record(row: dict, sentences: list[dict], entities: list[dict]) -> dict:
    text = row["text"]
    if len(sentences) != integer(row, "sentence_count"):
        raise ValueError(f"Sentence count mismatch for {row['example_id']}")
    if len(entities) != integer(row, "entity_count"):
        raise ValueError(f"Entity count mismatch for {row['example_id']}")
    for sentence in sentences:
        if text[sentence["start_char"] : sentence["end_char"]] != sentence["text"]:
            raise ValueError(f"Sentence offset mismatch for {row['example_id']}")
    for entity in entities:
        if text[entity["start_char"] : entity["end_char"]] != entity["text"]:
            raise ValueError(f"Entity offset mismatch for {row['example_id']}")

    return {
        "doc_id": row["doc_id"],
        "entities": entities,
        "example_id": row["example_id"],
        "length_bucket": row["length_bucket"],
        "schema_version": "1.0",
        "sections": [],
        "sentence_count": integer(row, "sentence_count"),
        "sentences": sentences,
        "source_dataset": "scirex",
        "source_document_sentence_count": integer(row, "source_document_sentence_count"),
        "source_sentence_end_exclusive": integer(row, "source_sentence_end_exclusive"),
        "source_sentence_start": integer(row, "source_sentence_start"),
        "source_split": row["source_split"],
        "source_token_end_exclusive": integer(row, "source_token_end_exclusive"),
        "source_token_start": integer(row, "source_token_start"),
        "text": text,
    }


def main() -> None:
    args = parse_args()
    examples_path = args.csv_dir / "examples_1000.csv"
    sentences_path = args.csv_dir / "sentences_1000.csv"
    entities_path = args.csv_dir / "entities_1000.csv"
    for path in (examples_path, sentences_path, entities_path):
        if not path.exists():
            raise FileNotFoundError(f"Required public CSV not found: {path}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists; pass --overwrite to replace it")

    sentences = grouped_sentences(sentences_path)
    entities = grouped_entities(entities_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    count = 0
    try:
        with examples_path.open(encoding="utf-8", newline="") as source:
            with temporary.open("w", encoding="utf-8", newline="\n") as target:
                for row in csv.DictReader(source):
                    example_id = row["example_id"]
                    record = build_record(
                        row,
                        sentences.pop(example_id, []),
                        entities.pop(example_id, []),
                    )
                    target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += 1
        if sentences or entities:
            raise ValueError("Sentence or entity CSV contains unknown example IDs")
        temporary.replace(args.output)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"Rebuilt {count} validated SciREX examples at {args.output}")
    print("Public CSVs omit section spans, so sections are emitted as an empty list.")


if __name__ == "__main__":
    main()
