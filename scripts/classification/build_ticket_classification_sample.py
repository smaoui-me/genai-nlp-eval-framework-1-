"""
Build a cleaned classification eval dataset from raw customer support tickets.

Outputs:
  data/classification/processed/ticket_extraction_eval.csv
  data/classification/processed/ticket_allowed_labels.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/classification/raw/customer_support_tickets.csv")
PROCESSED_CSV = Path("data/classification/processed/ticket_extraction_eval.csv")
ALLOWED_LABELS_PATH = Path("data/classification/processed/ticket_allowed_labels.json")

TAG_COLS = [f"tag_{i}" for i in range(1, 9)]


def load_and_clean(path: Path) -> pd.DataFrame:
    """Load raw CSV and do basic cleaning."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=["subject", "body"])
    df["subject"] = df["subject"].str.strip()
    df["body"] = df["body"].str.strip()
    return df


def build_text(df: pd.DataFrame) -> pd.Series:
    """Combine subject and body into a single text field."""
    return df["subject"] + "\n\n" + df["body"]


def collect_tags(df: pd.DataFrame) -> pd.Series:
    """Collect tag_1..tag_8 into a deduplicated list, dropping blanks."""
    def row_tags(row):
        tags = []
        for col in TAG_COLS:
            val = row.get(col, None)
            if pd.notna(val) and str(val).strip():
                tag = str(val).strip()
                if tag not in tags:
                    tags.append(tag)
        return tags

    return df.apply(row_tags, axis=1)


def build_allowed_labels(df: pd.DataFrame) -> dict:
    """Extract sorted unique values for types, queues, and tags."""
    types = sorted(df["type"].dropna().str.strip().unique().tolist())
    queues = sorted(df["queue"].dropna().str.strip().unique().tolist())

    all_tags = set()
    for col in TAG_COLS:
        if col in df.columns:
            vals = df[col].dropna().str.strip()
            all_tags.update(v for v in vals if v)

    return {
        "types": types,
        "queues": queues,
        "tags": sorted(all_tags),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the cleaned ticket-classification dataset.")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {RAW_PATH} ...")
    df = load_and_clean(RAW_PATH)

    df["ticket_id"] = range(1, len(df) + 1)
    df["text"] = build_text(df)
    df["gold_type"] = df["type"].str.strip()
    df["gold_queue"] = df["queue"].str.strip()
    df["gold_tags"] = collect_tags(df).apply(json.dumps)

    out = df[["ticket_id", "text", "gold_type", "gold_queue", "gold_tags"]].head(args.limit)
    out.to_csv(PROCESSED_CSV, index=False, lineterminator="\n")
    print(f"Saved {len(out)} rows to {PROCESSED_CSV}")

    allowed = build_allowed_labels(df)
    ALLOWED_LABELS_PATH.write_text(
        json.dumps(allowed, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Saved allowed labels to {ALLOWED_LABELS_PATH}")
    print(f"  {len(allowed['types'])} types, {len(allowed['queues'])} queues, {len(allowed['tags'])} tags")


if __name__ == "__main__":
    main()
