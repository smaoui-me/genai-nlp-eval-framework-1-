"""
Run the canonical ticket classification-with-evidence pipeline.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from genai_eval.classification.methods import METHOD_REGISTRY
from genai_eval.label_candidates import build_tag_frequency, parse_tag_list, select_candidate_tags


def load_allowed_labels(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def get_column_mapping(config: dict) -> dict:
    return config.get("dataset", {}).get(
        "columns",
        {
            "ticket_id": "ticket_id",
            "text": "text",
            "gold_type": "gold_type",
            "gold_queue": "gold_queue",
            "gold_tags": "gold_tags",
        },
    )


def validate_columns(df: pd.DataFrame, columns: dict, require_gold: bool = True) -> None:
    required_fields = ["text"]
    if require_gold:
        required_fields.extend(["gold_type", "gold_queue", "gold_tags"])
    missing = [columns[field] for field in required_fields if columns[field] not in df.columns]
    if missing:
        raise KeyError(f"Missing dataset columns: {missing}")


def run_pipeline(df: pd.DataFrame, method, allowed_labels: dict, columns: dict, candidate_cfg: dict, tag_frequency: dict[str, int], output_path: Path) -> list[dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    max_candidates = int(candidate_cfg.get("max_candidates", 40))
    fallback_top_k = int(candidate_cfg.get("fallback_top_k", 20))

    with output_path.open("w", encoding="utf-8") as fout:
        total_rows = len(df)
        for row_index, (_, row) in enumerate(df.iterrows(), start=1):
            ticket_id = row.get(columns["ticket_id"], row_index)
            text = str(row[columns["text"]])
            gold_type = str(row.get(columns["gold_type"], ""))
            gold_queue = str(row.get(columns["gold_queue"], ""))
            gold_tags = parse_tag_list(row.get(columns["gold_tags"], "[]"))
            candidate_tags = select_candidate_tags(
                text=text,
                allowed_tags=allowed_labels["tags"],
                tag_frequency=tag_frequency,
                max_candidates=max_candidates,
                fallback_top_k=fallback_top_k,
            )
            print(f"[{row_index}/{total_rows}] ticket_id={ticket_id} calling {method.name}", flush=True)
            try:
                result = method.extract_record(
                    text,
                    allowed_labels,
                    context={"candidate_tags": candidate_tags, "ticket_id": ticket_id},
                )
                print(
                    f"[{row_index}/{total_rows}] ticket_id={ticket_id} done "
                    f"(json_valid={result['json_validity']['all_json_valid']}, tags={len(result['validated_output']['tags'])})",
                    flush=True,
                )
                error_msg = None
            except Exception as exc:  # noqa: BLE001
                result = {
                    "raw_responses": {},
                    "parsed_output": {},
                    "validated_output": {
                        "type": {"label": "", "evidence": ""},
                        "queue": {"label": "", "evidence": ""},
                        "tags": [],
                    },
                    "json_validity": {"all_json_valid": False, "runtime_error": str(exc)},
                    "validation": {
                        "has_invalid_labels": False,
                        "invalid_labels": {"type": [], "queue": [], "tags": []},
                        "tags_outside_candidates": [],
                    },
                }
                error_msg = f"{type(exc).__name__}: {exc}"
                print(f"[{row_index}/{total_rows}] ticket_id={ticket_id} failed: {error_msg}", flush=True)

            record = {
                "ticket_id": ticket_id,
                "method": method.name,
                "model": getattr(method, "model_name", ""),
                "text": text,
                "candidate_tags": candidate_tags,
                "gold": {"type": gold_type, "queue": gold_queue, "tags": gold_tags},
                "raw_responses": result.get("raw_responses", {}),
                "parsed_output": result.get("parsed_output", {}),
                "prediction": result["validated_output"],
                "validated_output": result["validated_output"],
                "json_validity": result.get("json_validity", {}),
                "validation": result.get("validation", {}),
                "retrieval": result.get("retrieval"),
            }
            if error_msg:
                record["error"] = error_msg
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

    print(f"Saved classification records to {output_path}")
    return records


def save_evaluation(records: list[dict], scores_path: Path, errors_path: Path) -> None:
    # Keep prediction-only deployments independent of evaluation-only packages.
    from genai_eval.classification.evaluator import build_error_rows, evaluate_predictions

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([evaluate_predictions(records)]).to_csv(scores_path, index=False)
    pd.DataFrame(build_error_rows(records)).to_csv(errors_path, index=False)
    print(f"Saved scores to {scores_path}")
    print(f"Saved errors to {errors_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ticket classification with evidence.")
    parser.add_argument("--method", required=True, choices=list(METHOD_REGISTRY.keys()))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Allow unlabeled input and skip metric calculation.",
    )
    args = parser.parse_args()

    method_cls = METHOD_REGISTRY[args.method]
    method = method_cls(Path(args.config)) if args.config else method_cls()
    config = getattr(method, "config", {})
    dataset_cfg = config.get("dataset", {})
    input_path = Path(args.input or dataset_cfg.get("input_path", "data/classification/processed/ticket_extraction_eval.csv"))
    labels_path = Path(args.labels or dataset_cfg.get("labels_path", "data/classification/processed/ticket_allowed_labels.json"))

    full_df = load_dataframe(input_path)
    columns = get_column_mapping(config)
    validate_columns(full_df, columns, require_gold=not args.predict_only)
    limit = args.limit if args.limit is not None else config.get("debug", {}).get("max_rows", 20)
    df = full_df.head(limit) if limit and limit > 0 else full_df.copy()
    allowed_labels = load_allowed_labels(labels_path)
    tag_frequency = (
        build_tag_frequency(full_df[columns["gold_tags"]].tolist())
        if columns["gold_tags"] in full_df.columns
        else {}
    )

    run_name = args.run_name or method.name
    output_path = Path(args.output or f"results/classification/{run_name}.jsonl")
    scores_path = Path(f"results/classification/evaluation/{run_name}_scores.csv")
    errors_path = Path(f"results/classification/evaluation/{run_name}_errors.csv")

    print(f"Running method `{args.method}` on {len(df)} rows")
    records = run_pipeline(df, method, allowed_labels, columns, config.get("candidate_tags", {}), tag_frequency, output_path)
    if args.predict_only:
        print("Prediction-only mode: skipped metrics because no independent gold labels were required")
    else:
        save_evaluation(records, scores_path, errors_path)


if __name__ == "__main__":
    main()
