"""Run few-shot free-form location extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import genai_eval.extraction.evaluator as extraction_evaluator
from genai_eval.extraction.methods import METHOD_REGISTRY
from run_common import load_dataframe, run_pipeline, save_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run few-shot free-form extraction.")
    parser.add_argument("--method", default="few_shot_freeform", choices=["few_shot_freeform"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    method_cls = METHOD_REGISTRY[args.method]
    method = method_cls(Path(args.config)) if args.config else method_cls()
    config = getattr(method, "config", {})
    dataset_cfg = config.get("dataset", {})
    input_path = Path(args.input or dataset_cfg.get("input_path", "data/extraction/processed/fewnerd_location_test_500.csv"))
    full_df = load_dataframe(input_path)
    limit = args.limit if args.limit is not None else config.get("debug", {}).get("max_rows", 100)
    df = full_df.head(limit) if limit and limit > 0 else full_df.copy()
    run_name = args.run_name or method.name
    output_path = Path(args.output or f"results/extraction/{run_name}.jsonl")
    scores_path = Path(f"results/extraction/evaluation/{run_name}_scores.csv")
    rows_path = Path(f"results/extraction/evaluation/{run_name}_rows.csv")
    print(f"Running method `{args.method}` on {len(df)} rows")
    records = run_pipeline(df, method, output_path)
    save_evaluation(records, extraction_evaluator, scores_path, rows_path)


if __name__ == "__main__":
    main()
