"""Run the canonical zero-shot extraction pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from genai_eval.extraction.evaluator import build_error_rows, build_per_type_rows, evaluate_predictions
from genai_eval.extraction.methods import METHOD_REGISTRY


def load_allowed_entity_types(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("entity_types", [])


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def parse_json_list(value: str):
    if isinstance(value, str):
        return json.loads(value)
    return value


def run_pipeline(df: pd.DataFrame, method, allowed_entity_types: list[str], output_path: Path) -> list[dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with output_path.open("w", encoding="utf-8") as fout:
        total_rows = len(df)
        for row_index, (_, row) in enumerate(df.iterrows(), start=1):
            sample_id = row["id"]
            tokens = parse_json_list(row["tokens"])
            gold_spans = parse_json_list(row["spans"])
            text = " ".join(tokens)
            print(f"[{row_index}/{total_rows}] id={sample_id} calling {method.name}", flush=True)
            try:
                result = method.extract_record(tokens=tokens, allowed_entity_types=allowed_entity_types)
                print(
                    f"[{row_index}/{total_rows}] id={sample_id} done "
                    f"(json_valid={result['json_validity']['all_json_valid']}, entities={len(result['validated_output']['entities'])})",
                    flush=True,
                )
                error_msg = None
            except Exception as exc:  # noqa: BLE001
                result = {
                    "raw_responses": {},
                    "parsed_output": {},
                    "validated_output": {"entities": []},
                    "json_validity": {"all_json_valid": False, "runtime_error": str(exc)},
                    "validation": {"has_invalid_labels": False, "invalid_entities": []},
                }
                error_msg = f"{type(exc).__name__}: {exc}"
                print(f"[{row_index}/{total_rows}] id={sample_id} failed: {error_msg}", flush=True)

            record = {
                "id": sample_id,
                "method": method.name,
                "model": getattr(method, "model_name", ""),
                "tokens": tokens,
                "text": text,
                "gold": {"spans": gold_spans},
                "raw_responses": result.get("raw_responses", {}),
                "parsed_output": result.get("parsed_output", {}),
                "prediction": result["validated_output"],
                "validated_output": result["validated_output"],
                "json_validity": result.get("json_validity", {}),
                "validation": result.get("validation", {}),
            }
            if error_msg:
                record["error"] = error_msg
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
    print(f"Saved extraction records to {output_path}")
    return records


def save_evaluation(records: list[dict], scores_path: Path, errors_path: Path, per_type_path: Path) -> None:
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([evaluate_predictions(records)]).to_csv(scores_path, index=False)
    pd.DataFrame(build_error_rows(records)).to_csv(errors_path, index=False)
    pd.DataFrame(build_per_type_rows(records)).to_csv(per_type_path, index=False)
    print(f"Saved scores to {scores_path}")
    print(f"Saved errors to {errors_path}")
    print(f"Saved per-type scores to {per_type_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-shot extraction.")
    parser.add_argument("--method", default="zero_shot", choices=list(METHOD_REGISTRY.keys()))
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    method_cls = METHOD_REGISTRY[args.method]
    method = method_cls(Path(args.config)) if args.config else method_cls()
    config = getattr(method, "config", {})
    dataset_cfg = config.get("dataset", {})
    input_path = Path(args.input or dataset_cfg.get("input_path", "data/extraction/processed/fewnerd_intra_test.csv"))
    labels_path = Path(args.labels or dataset_cfg.get("labels_path", "data/extraction/processed/fewnerd_allowed_entity_types.json"))

    full_df = load_dataframe(input_path)
    limit = args.limit if args.limit is not None else config.get("debug", {}).get("max_rows", 20)
    df = full_df.head(limit) if limit and limit > 0 else full_df.copy()
    allowed_entity_types = load_allowed_entity_types(labels_path)

    run_name = args.run_name or method.name
    output_path = Path(args.output or f"results/extraction/{run_name}.jsonl")
    scores_path = Path(f"results/extraction/evaluation/{run_name}_scores.csv")
    errors_path = Path(f"results/extraction/evaluation/{run_name}_errors.csv")
    per_type_path = Path(f"results/extraction/evaluation/{run_name}_per_type.csv")

    print(f"Running method `{args.method}` on {len(df)} rows")
    records = run_pipeline(df, method, allowed_entity_types, output_path)
    save_evaluation(records, scores_path, errors_path, per_type_path)


if __name__ == "__main__":
    main()
