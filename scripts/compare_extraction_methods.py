"""
compare_extraction_methods.py

Reads all *_scores.csv files from results/evaluation/ and combines them
into a single comparison table: ticket_extraction_method_comparison.csv.

Usage:
    python scripts/compare_extraction_methods.py
    python scripts/compare_extraction_methods.py --eval-dir results/evaluation
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


COMPARISON_COLUMNS = [
    "method",
    "type_accuracy",
    "queue_accuracy",
    "tag_micro_f1",
    "tag_row_f1",
    "evidence_valid_rate",
]

OUTPUT_FILENAME = "ticket_extraction_method_comparison.csv"


def load_score_files(eval_dir: Path) -> list:
    """Find and load all *_scores.csv files in the evaluation directory.

    Excludes the comparison file itself to avoid re-reading on repeated runs.

    Args:
        eval_dir: Path to the evaluation results directory.

    Returns:
        List of DataFrames, one per score file.
    """
    score_files = [
        p for p in sorted(eval_dir.glob("*_scores.csv"))
        if p.name != OUTPUT_FILENAME
    ]

    if not score_files:
        print(f"No *_scores.csv files found in {eval_dir}")
        return []

    dfs = []
    for path in score_files:
        df = pd.read_csv(path)
        print(f"  Loaded {path.name} ({len(df)} row(s))")
        dfs.append(df)
    return dfs


def build_comparison(dfs: list) -> pd.DataFrame:
    """Combine score DataFrames and select comparison columns.

    Missing columns are filled with None so the output always has
    the full set of comparison columns.

    Args:
        dfs: List of score DataFrames.

    Returns:
        Combined comparison DataFrame.
    """
    combined = pd.concat(dfs, ignore_index=True)
    for col in COMPARISON_COLUMNS:
        if col not in combined.columns:
            combined[col] = None
    return combined[COMPARISON_COLUMNS].sort_values("tag_micro_f1", ascending=False)


def main():
    parser = argparse.ArgumentParser(
        description="Compare ticket extraction method scores."
    )
    parser.add_argument(
        "--eval-dir",
        default="results/evaluation",
        help="Directory containing *_scores.csv files",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"Error: evaluation directory not found: {eval_dir}")
        sys.exit(1)

    print(f"Reading score files from {eval_dir} ...")
    dfs = load_score_files(eval_dir)
    if not dfs:
        sys.exit(1)

    comparison = build_comparison(dfs)
    output_path = eval_dir / OUTPUT_FILENAME
    comparison.to_csv(output_path, index=False)
    print(f"\nComparison saved to {output_path}")
    print("\n" + comparison.to_string(index=False))


if __name__ == "__main__":
    main()
