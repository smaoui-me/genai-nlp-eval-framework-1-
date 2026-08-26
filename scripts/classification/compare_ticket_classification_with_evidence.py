"""
Compare classification-with-evidence method score files.
"""

import argparse
from pathlib import Path

import pandas as pd


COMPARISON_COLUMNS = [
    "method",
    "model",
    "type_accuracy",
    "type_macro_f1",
    "queue_accuracy",
    "queue_macro_f1",
    "tag_micro_f1",
    "tag_micro_precision",
    "tag_micro_recall",
    "evidence_valid_rate",
]


def main():
    parser = argparse.ArgumentParser(description="Compare classification-with-evidence methods.")
    parser.add_argument("--eval-dir", default="results/classification/evaluation")
    parser.add_argument("--pattern", default="*_scores.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    score_files = sorted(eval_dir.glob(args.pattern))
    if not score_files:
        print(f"No {args.pattern!r} files found in {eval_dir}")
        return

    combined = pd.concat([pd.read_csv(path) for path in score_files], ignore_index=True)
    for column in COMPARISON_COLUMNS:
        if column not in combined.columns:
            combined[column] = None
    output_path = Path(args.output) if args.output else eval_dir / "classification_method_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined[COMPARISON_COLUMNS].sort_values("tag_micro_f1", ascending=False).to_csv(output_path, index=False)
    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()
