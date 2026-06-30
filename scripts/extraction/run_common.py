"""Shared helpers for extraction run scripts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ALLOWED_ENTITY_TYPES = ["location"]


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def parse_json_list(value: str):
    return json.loads(value) if isinstance(value, str) else value


def run_pipeline(df: pd.DataFrame, method, output_path: Path) -> list[dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with output_path.open("w", encoding="utf-8") as fout:
        total_rows = len(df)
        for row_index, (_, row) in enumerate(df.iterrows(), start=1):
            sample_id = row["id"]
            sentence = str(row["sentence"])
            tokens = parse_json_list(row["tokens"])
            gold_entities = parse_json_list(row["gold_entities"])
            print(f"[{row_index}/{total_rows}] id={sample_id} calling {method.name}", flush=True)
            try:
                result = method.extract_record(sentence=sentence, tokens=tokens, allowed_entity_types=ALLOWED_ENTITY_TYPES)
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
                "text": sentence,
                "gold": {"spans": gold_entities},
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


def save_evaluation(records: list[dict], evaluator_module, scores_path: Path, rows_path: Path) -> None:
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([evaluator_module.evaluate_predictions(records)]).to_csv(scores_path, index=False)
    pd.DataFrame(evaluator_module.build_error_rows(records)).to_csv(rows_path, index=False)
    print(f"Saved scores to {scores_path}")
    print(f"Saved per-sentence breakdown to {rows_path}")
