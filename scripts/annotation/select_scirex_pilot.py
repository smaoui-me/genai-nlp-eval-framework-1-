"""Select a deterministic, document-disjoint SciREX development pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BUCKETS = ("short", "medium", "long", "very_long")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select(records: list[dict], per_bucket: int = 5, split: str = "dev") -> list[dict]:
    used_docs: set[str] = set()
    selected: list[dict] = []
    for bucket in BUCKETS:
        candidates = sorted(
            (row for row in records if (split == "all" or row["source_split"] == split)
             and row["length_bucket"] == bucket),
            key=lambda row: row["example_id"],
        )
        chosen = []
        for row in candidates:
            if row["doc_id"] in used_docs:
                continue
            chosen.append(row)
            used_docs.add(row["doc_id"])
            if len(chosen) == per_bucket:
                break
        if len(chosen) != per_bucket:
            raise ValueError(f"Could select only {len(chosen)} document-disjoint {bucket} examples")
        selected.extend(chosen)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/annotation/processed/scirex/dev.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("eval/corpora/scirex_dev_pilot_20.jsonl"))
    parser.add_argument("--per-bucket", type=int, default=5)
    parser.add_argument("--split", choices=("train", "dev", "test", "all"), default="dev")
    parser.add_argument("--exclude-manifest", type=Path, help="Exclude every source paper listed here")
    args = parser.parse_args()
    records = read_jsonl(args.input)
    if args.exclude_manifest:
        excluded_docs = {row["doc_id"] for row in read_jsonl(args.exclude_manifest)}
        records = [row for row in records if row["doc_id"] not in excluded_docs]
    selected = select(records, args.per_bucket, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"Wrote {len(selected)} {args.split} examples from {len({x['doc_id'] for x in selected})} papers")
    print(f"Full texts contain {sum(x['sentence_count'] for x in selected)} sentences and "
          f"{sum(len(x['entities']) for x in selected)} gold entities")


if __name__ == "__main__":
    main()
