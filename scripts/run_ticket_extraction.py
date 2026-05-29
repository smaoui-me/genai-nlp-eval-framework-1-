"""
run_ticket_extraction.py

Runs a selected extraction method on the processed ticket evaluation dataset.
Saves results as JSONL with gold labels and predictions side-by-side.

Usage:
    python scripts/run_ticket_extraction.py --method zero_shot --limit 50
    python scripts/run_ticket_extraction.py --method embedding --limit 50
    python scripts/run_ticket_extraction.py --method few_shot
"""

import json
import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genai_eval.extraction_methods import METHOD_REGISTRY


def load_dataset(path: Path, limit: int) -> pd.DataFrame:
    """Load the processed evaluation CSV and optionally limit rows."""
    df = pd.read_csv(path)
    if limit and limit > 0:
        df = df.head(limit)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def load_allowed_labels(path: Path) -> dict:
    """Load the allowed labels JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def parse_gold_tags(gold_tags_str: str) -> list:
    """Parse the gold_tags JSON string from the CSV into a list."""
    try:
        return json.loads(gold_tags_str)
    except (json.JSONDecodeError, TypeError):
        return []


def run_extraction(df: pd.DataFrame, method, allowed_labels: dict, output_path: Path):
    """Run extraction on every row and write results to a JSONL file.

    Args:
        df: Processed tickets dataframe.
        method: Instantiated ExtractionMethod object.
        allowed_labels: Dict with keys "types", "queues", "tags".
        output_path: Path to write output JSONL.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_err = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for _, row in df.iterrows():
            ticket_id = int(row["ticket_id"])
            text = str(row["text"])
            gold_type = str(row.get("gold_type", ""))
            gold_queue = str(row.get("gold_queue", ""))
            gold_tags = parse_gold_tags(row.get("gold_tags", "[]"))

            try:
                prediction = method.extract(text, allowed_labels)
                error_msg = None
                n_ok += 1
            except NotImplementedError as e:
                prediction = {
                    "type": {"label": "", "evidence": ""},
                    "queue": {"label": "", "evidence": ""},
                    "tags": [],
                }
                error_msg = f"NotImplementedError: {e}"
                n_err += 1
            except Exception as e:  # noqa: BLE001
                prediction = {
                    "type": {"label": "", "evidence": ""},
                    "queue": {"label": "", "evidence": ""},
                    "tags": [],
                }
                error_msg = f"{type(e).__name__}: {e}"
                n_err += 1

            record = {
                "ticket_id": ticket_id,
                "method": method.name,
                "text": text,
                "gold": {
                    "type": gold_type,
                    "queue": gold_queue,
                    "tags": gold_tags,
                },
                "prediction": prediction,
            }
            if error_msg:
                record["error"] = error_msg

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (n_ok + n_err) % 10 == 0:
                print(f"  Processed {n_ok + n_err} tickets ...")

    print(f"Done. {n_ok} succeeded, {n_err} failed. Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ticket extraction method.")
    parser.add_argument(
        "--method",
        required=True,
        choices=list(METHOD_REGISTRY.keys()),
        help="Extraction method to run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of tickets to process (0 = all)",
    )
    parser.add_argument(
        "--input",
        default="data/processed/ticket_extraction_eval.csv",
        help="Path to processed evaluation CSV",
    )
    parser.add_argument(
        "--labels",
        default="data/processed/ticket_allowed_labels.json",
        help="Path to allowed labels JSON",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: results/extractions/{method}_ticket_extraction.jsonl)",
    )
    args = parser.parse_args()

    output_path = Path(
        args.output or f"results/extractions/{args.method}_ticket_extraction.jsonl"
    )
    input_path = Path(args.input)
    labels_path = Path(args.labels)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)
    if not labels_path.exists():
        print(f"Error: labels file not found: {labels_path}")
        sys.exit(1)

    df = load_dataset(input_path, args.limit)
    allowed_labels = load_allowed_labels(labels_path)

    method_cls = METHOD_REGISTRY[args.method]
    method = method_cls()
    print(f"Running method: {method.name}")

    run_extraction(df, method, allowed_labels, output_path)


if __name__ == "__main__":
    main()
