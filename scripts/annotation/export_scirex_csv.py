"""Export normalized SciREX benchmark JSONL into linked, flat CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write_csv(path: Path, fieldnames: list[str], rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc


def export(input_path: Path, output_dir: Path) -> dict[str, int]:
    examples = list(read_jsonl(input_path))
    example_fields = [
        "example_id", "doc_id", "source_split", "length_bucket", "sentence_count",
        "source_document_sentence_count", "source_sentence_start",
        "source_sentence_end_exclusive", "source_token_start",
        "source_token_end_exclusive", "character_count", "entity_count",
        "section_count", "text",
    ]
    example_count = write_csv(
        output_dir / "examples_1000.csv", example_fields,
        ({
            **{field: item.get(field) for field in example_fields if field not in {
                "character_count", "entity_count", "section_count",
            }},
            "character_count": len(item["text"]),
            "entity_count": len(item["entities"]),
            "section_count": len(item["sections"]),
        } for item in examples),
    )

    sentence_fields = [
        "example_id", "doc_id", "relative_sentence_index", "source_sentence_index",
        "start_char", "end_char", "text",
    ]
    sentence_count = write_csv(
        output_dir / "sentences_1000.csv", sentence_fields,
        ({
            "example_id": item["example_id"], "doc_id": item["doc_id"], **sentence,
        } for item in examples for sentence in item["sentences"]),
    )

    entity_fields = [
        "example_id", "doc_id", "entity_id", "label", "text", "start_char",
        "end_char", "source_token_start", "source_token_end_exclusive",
    ]
    entity_count = write_csv(
        output_dir / "entities_1000.csv", entity_fields,
        ({
            "example_id": item["example_id"], "doc_id": item["doc_id"], **entity,
        } for item in examples for entity in item["entities"]),
    )
    return {"examples": example_count, "sentences": sentence_count, "entities": entity_count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/annotation/processed/scirex/examples_1000.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/annotation/processed/scirex/csv"),
    )
    args = parser.parse_args()
    counts = export(args.input, args.output_dir)
    print(f"Wrote {counts['examples']} examples, {counts['sentences']} sentences, "
          f"and {counts['entities']} entities to {args.output_dir}")


if __name__ == "__main__":
    main()
