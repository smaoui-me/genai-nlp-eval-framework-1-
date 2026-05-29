"""
build_ticket_extraction_sample.py

Loads raw customer support tickets CSV, builds processed evaluation dataset
with gold labels for type, queue, and tags. Also extracts allowed label sets.

Usage:
    python scripts/build_ticket_extraction_sample.py
"""

import json
import argparse
import pandas as pd
from pathlib import Path


TAG_COLUMNS = [f"tag_{i}" for i in range(1, 9)]


def load_raw_tickets(path: Path) -> pd.DataFrame:
    """Load raw ticket CSV from disk."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def build_text(df: pd.DataFrame) -> pd.Series:
    """Concatenate subject and body into a single input text."""
    subject = df["subject"].fillna("").str.strip()
    body = df["body"].fillna("").str.strip()
    return subject + "\n\n" + body


def collect_tags(row: pd.Series) -> list:
    """Combine tag_1 through tag_8 into a list, dropping empty values."""
    tags = []
    for col in TAG_COLUMNS:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            tags.append(str(val).strip())
    return tags


def build_allowed_labels(df: pd.DataFrame) -> dict:
    """Extract unique label sets for types, queues, and tags."""
    types = sorted(df["type"].dropna().unique().tolist())
    queues = sorted(df["queue"].dropna().unique().tolist())

    tag_set = set()
    for _, row in df.iterrows():
        for col in TAG_COLUMNS:
            val = row.get(col, "")
            if pd.notna(val) and str(val).strip():
                tag_set.add(str(val).strip())
    tags = sorted(tag_set)

    return {"types": types, "queues": queues, "tags": tags}


def process_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw dataframe into evaluation-ready format."""
    out = pd.DataFrame()
    out["ticket_id"] = range(1, len(df) + 1)
    out["text"] = build_text(df)
    out["gold_type"] = df["type"].fillna("").str.strip().values
    out["gold_queue"] = df["queue"].fillna("").str.strip().values
    out["gold_tags"] = df.apply(lambda row: json.dumps(collect_tags(row)), axis=1)
    return out


def main():
    parser = argparse.ArgumentParser(description="Build ticket extraction eval sample.")
    parser.add_argument(
        "--input",
        default="data/raw/customer_support_tickets.csv",
        help="Path to raw tickets CSV",
    )
    parser.add_argument(
        "--output-data",
        default="data/processed/ticket_extraction_eval.csv",
        help="Output path for processed CSV",
    )
    parser.add_argument(
        "--output-labels",
        default="data/processed/ticket_allowed_labels.json",
        help="Output path for allowed labels JSON",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_data = Path(args.output_data)
    output_labels = Path(args.output_labels)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_data.parent.mkdir(parents=True, exist_ok=True)
    output_labels.parent.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_tickets(input_path)

    df_processed = process_tickets(df_raw)
    df_processed.to_csv(output_data, index=False)
    print(f"Saved processed data ({len(df_processed)} rows) to {output_data}")

    allowed_labels = build_allowed_labels(df_raw)
    output_labels.write_text(json.dumps(allowed_labels, indent=2))
    print(f"Saved allowed labels to {output_labels}")
    print(
        f"  types: {len(allowed_labels['types'])}, "
        f"queues: {len(allowed_labels['queues'])}, "
        f"tags: {len(allowed_labels['tags'])}"
    )


if __name__ == "__main__":
    main()
