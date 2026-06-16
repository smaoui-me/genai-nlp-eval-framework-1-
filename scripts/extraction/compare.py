"""Compare extraction score files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COMPARISON_COLUMNS = [
    "method",
    "model",
    "strict_f1",
    "strict_precision",
    "strict_recall",
    "lenient_f1",
    "lenient_precision",
    "lenient_recall",
    "json_valid_rate",
    "invalid_label_rate",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare extraction methods.")
    parser.add_argument("--eval-dir", default="results/extraction/evaluation")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    score_files = sorted(eval_dir.glob("*_scores.csv"))
    if not score_files:
        print(f"No *_scores.csv files found in {eval_dir}")
        return

    combined = pd.concat([pd.read_csv(path) for path in score_files], ignore_index=True)
    for column in COMPARISON_COLUMNS:
        if column not in combined.columns:
            combined[column] = None
    output_path = eval_dir / "extraction_method_comparison.csv"
    combined[COMPARISON_COLUMNS].sort_values("strict_f1", ascending=False).to_csv(output_path, index=False)
    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()
