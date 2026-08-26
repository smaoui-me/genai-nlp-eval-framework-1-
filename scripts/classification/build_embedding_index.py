"""Build a reproducible retrieval index from reviewed classification examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from genai_eval.classification.retrieval.embedding_index import encode_texts, load_encoder, write_index
from genai_eval.label_candidates import parse_tag_list


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index labeled reference tickets for embedding-RAG classification."
    )
    parser.add_argument("--input", type=Path, required=True, help="Reviewed reference CSV; do not use the test set.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/classification/retrieval/index"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--id-column", default="ticket_id")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--type-column", default="gold_type")
    parser.add_argument("--queue-column", default="gold_queue")
    parser.add_argument("--tags-column", default="gold_tags")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")

    columns = {
        "ticket_id": args.id_column,
        "text": args.text_column,
        "gold_type": args.type_column,
        "gold_queue": args.queue_column,
        "gold_tags": args.tags_column,
    }
    frame = pd.read_csv(args.input)
    missing = [name for name in columns.values() if name not in frame.columns]
    if missing:
        raise KeyError(f"Reference CSV is missing columns: {missing}")
    if args.limit:
        frame = frame.head(args.limit)

    records = []
    for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
        required_values = [
            row[args.id_column],
            row[args.text_column],
            row[args.type_column],
            row[args.queue_column],
        ]
        if any(pd.isna(value) for value in required_values):
            raise ValueError(f"Reference row {row_number} has a missing ID, text, type, or queue")
        text = str(row[args.text_column]).strip()
        type_label = str(row[args.type_column]).strip()
        queue_label = str(row[args.queue_column]).strip()
        if not text or not type_label or not queue_label:
            raise ValueError(f"Reference row {row_number} has empty text, type, or queue")
        records.append(
            {
                "ticket_id": str(row[args.id_column]),
                "text": text,
                "type": type_label,
                "queue": queue_label,
                "tags": parse_tag_list(row[args.tags_column]),
            }
        )
    if len({record["ticket_id"] for record in records}) != len(records):
        raise ValueError("Reference ticket IDs must be unique")

    print(f"Encoding {len(records)} reviewed examples with {args.model}")
    encoder = load_encoder(args.model)
    embeddings = encode_texts(encoder, [record["text"] for record in records], args.batch_size)
    manifest = write_index(args.output_dir, records, embeddings, args.model, args.input, columns)

    labels = {
        "types": sorted({record["type"] for record in records}),
        "queues": sorted({record["queue"] for record in records}),
        "tags": sorted({tag for record in records for tag in record["tags"]}),
    }
    labels_path = args.output_dir / "allowed_labels.json"
    labels_path.write_text(
        json.dumps(labels, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Saved index to {args.output_dir}: {manifest['record_count']} records, "
        f"{manifest['embedding_dimension']} dimensions"
    )
    print(f"Saved derived label schema to {labels_path}")


if __name__ == "__main__":
    main()
