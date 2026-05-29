"""
Run ticket extraction on the evaluation dataset and save predictions plus metrics.

Usage:
    python scripts/run_ticket_extraction.py --method zero_shot
    python scripts/run_ticket_extraction.py --method agent_two_step --limit 20
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genai_eval.extraction_methods import METHOD_REGISTRY
from genai_eval.extraction_methods.extraction_evaluator import (
    build_error_rows,
    evaluate_predictions,
)
from genai_eval.label_candidates import build_tag_frequency, parse_tag_list, select_candidate_tags


def load_allowed_labels(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_paths(args, config: dict) -> tuple[Path, Path]:
    dataset_cfg = config.get("dataset", {})
    input_path = Path(args.input or dataset_cfg.get("input_path", "data/processed/ticket_extraction_eval.csv"))
    labels_path = Path(args.labels or dataset_cfg.get("labels_path", "data/processed/ticket_allowed_labels.json"))
    return input_path, labels_path


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


def get_limit(args, config: dict) -> int | None:
    if args.limit is not None:
        return args.limit
    return config.get("debug", {}).get("max_rows", 20)


def build_output_paths(method_name: str, output_jsonl: str | None) -> tuple[Path, Path, Path]:
    extraction_path = Path(
        output_jsonl or f"results/extractions/{method_name}_ticket_extraction.jsonl"
    )
    scores_path = Path(f"results/evaluation/{method_name}_scores.csv")
    errors_path = Path(f"results/evaluation/{method_name}_errors.csv")
    return extraction_path, scores_path, errors_path


def build_output_paths_for_run(run_name: str, output_jsonl: str | None) -> tuple[Path, Path, Path]:
    extraction_path = Path(
        output_jsonl or f"results/extractions/{run_name}_ticket_extraction.jsonl"
    )
    scores_path = Path(f"results/evaluation/{run_name}_scores.csv")
    errors_path = Path(f"results/evaluation/{run_name}_errors.csv")
    return extraction_path, scores_path, errors_path


def validate_columns(df: pd.DataFrame, columns: dict) -> None:
    missing = [column_name for column_name in columns.values() if column_name not in df.columns]
    if missing:
        raise KeyError(f"Missing dataset columns: {missing}")


def run_extraction(
    df: pd.DataFrame,
    method,
    allowed_labels: dict,
    columns: dict,
    candidate_cfg: dict,
    tag_frequency: dict[str, int],
    output_path: Path,
) -> list[dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []

    max_candidates = int(candidate_cfg.get("max_candidates", 40))
    fallback_top_k = int(candidate_cfg.get("fallback_top_k", 20))
    retry_candidate_limits = list(candidate_cfg.get("retry_candidate_limits", []))

    with output_path.open("w", encoding="utf-8") as fout:
        total_rows = len(df)
        for row_index, (_, row) in enumerate(df.iterrows(), start=1):
            ticket_id = row[columns["ticket_id"]]
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

            print(
                f"[{row_index}/{total_rows}] ticket_id={ticket_id} "
                f"calling {method.name} with {len(candidate_tags)} candidate tags",
                flush=True,
            )

            attempt_limits = [len(candidate_tags)]
            attempt_limits.extend(
                limit for limit in retry_candidate_limits
                if isinstance(limit, int) and 0 < limit < len(candidate_tags)
            )

            method_result = None
            validated_output = None
            error_msg = None

            for attempt_index, candidate_limit in enumerate(attempt_limits, start=1):
                attempt_candidate_tags = candidate_tags[:candidate_limit]
                if attempt_index > 1:
                    print(
                        f"[{row_index}/{total_rows}] ticket_id={ticket_id} retrying "
                        f"with {candidate_limit} candidate tags",
                        flush=True,
                    )
                try:
                    method_result = method.extract_record(
                        text=text,
                        allowed_labels=allowed_labels,
                        context={"candidate_tags": attempt_candidate_tags},
                    )
                    validated_output = method_result["validated_output"]
                    candidate_tags = attempt_candidate_tags
                    error_msg = None
                    break
                except (InternalServerError, APIConnectionError, APITimeoutError, RateLimitError) as exc:
                    error_msg = f"{type(exc).__name__}: {exc}"
                    if attempt_index == len(attempt_limits):
                        method_result = {
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
                        validated_output = method_result["validated_output"]
                except Exception as exc:  # noqa: BLE001
                    error_msg = f"{type(exc).__name__}: {exc}"
                    method_result = {
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
                    validated_output = method_result["validated_output"]
                    break

            if error_msg and method_result is None:
                method_result = {
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
                validated_output = method_result["validated_output"]

            if error_msg:
                print(
                    f"[{row_index}/{total_rows}] ticket_id={ticket_id} failed: {error_msg}",
                    flush=True,
                )
            else:
                json_ok = method_result.get("json_validity", {}).get("all_json_valid", False)
                print(
                    f"[{row_index}/{total_rows}] ticket_id={ticket_id} done "
                    f"(json_valid={json_ok}, tags={len(validated_output['tags'])})",
                    flush=True,
                )

            record = {
                "ticket_id": ticket_id,
                "method": method.name,
                "model": getattr(method, "model_name", ""),
                "text": text,
                "candidate_tags": candidate_tags,
                "gold": {
                    "type": gold_type,
                    "queue": gold_queue,
                    "tags": gold_tags,
                },
                "raw_responses": method_result.get("raw_responses", {}),
                "parsed_output": method_result.get("parsed_output", {}),
                "prediction": validated_output,
                "validated_output": validated_output,
                "json_validity": method_result.get("json_validity", {}),
                "validation": method_result.get("validation", {}),
            }
            if error_msg:
                record["error"] = error_msg

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

    print(f"Saved extraction records to {output_path}")
    return records


def save_evaluation(records: list[dict], scores_path: Path, errors_path: Path) -> None:
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores = evaluate_predictions(records)
    pd.DataFrame([scores]).to_csv(scores_path, index=False)
    pd.DataFrame(build_error_rows(records)).to_csv(errors_path, index=False)
    print(f"Saved scores to {scores_path}")
    print(f"Saved errors to {errors_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ticket extraction.")
    parser.add_argument(
        "--method",
        required=True,
        choices=list(METHOD_REGISTRY.keys()),
        help="Extraction method to run",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max rows to process")
    parser.add_argument("--input", default=None, help="Optional CSV override")
    parser.add_argument("--labels", default=None, help="Optional labels JSON override")
    parser.add_argument("--output", default=None, help="Optional output JSONL override")
    parser.add_argument("--config", default=None, help="Optional method config override")
    parser.add_argument("--run-name", default=None, help="Versioned output prefix override")
    args = parser.parse_args()

    method_cls = METHOD_REGISTRY[args.method]
    method = method_cls(Path(args.config)) if args.config else method_cls()
    method.model_name = getattr(method, "model_name", "") or ""

    config = getattr(method, "config", {})
    input_path, labels_path = resolve_paths(args, config)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)
    if not labels_path.exists():
        print(f"Error: labels file not found: {labels_path}")
        sys.exit(1)

    full_df = load_dataframe(input_path)
    columns = get_column_mapping(config)
    validate_columns(full_df, columns)

    limit = get_limit(args, config)
    df = full_df.head(limit) if limit and limit > 0 else full_df.copy()
    allowed_labels = load_allowed_labels(labels_path)
    tag_frequency = build_tag_frequency(full_df[columns["gold_tags"]].tolist())
    run_name = args.run_name or Path(args.config).stem if args.config else method.name
    output_path, scores_path, errors_path = build_output_paths_for_run(run_name, args.output)

    print(f"Running method `{args.method}` on {len(df)} rows")
    try:
        records = run_extraction(
            df=df,
            method=method,
            allowed_labels=allowed_labels,
            columns=columns,
            candidate_cfg=config.get("candidate_tags", {}),
            tag_frequency=tag_frequency,
            output_path=output_path,
        )
    except KeyboardInterrupt:
        print("\nRun interrupted by user.", flush=True)
        sys.exit(130)

    save_evaluation(records, scores_path, errors_path)


if __name__ == "__main__":
    main()
