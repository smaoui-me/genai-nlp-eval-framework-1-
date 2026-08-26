import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    path = ROOT / "scripts" / "annotation" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_document_disjoint_balanced_selection():
    selector = load_script("select_scirex_pilot.py")
    records = []
    for bucket in selector.BUCKETS:
        for index in range(3):
            records.append({
                "example_id": f"{bucket}-{index}", "doc_id": f"{bucket}-doc-{index}",
                "source_split": "dev", "length_bucket": bucket,
            })
    selected = selector.select(records, per_bucket=2, split="dev")
    assert len(selected) == 8
    assert len({row["doc_id"] for row in selected}) == 8
    assert {bucket: sum(row["length_bucket"] == bucket for row in selected)
            for bucket in selector.BUCKETS} == {bucket: 2 for bucket in selector.BUCKETS}


def test_metrics_and_bootstrap_are_deterministic():
    evaluator = load_script("evaluate_scirex_batch.py")
    assert evaluator.metrics(5, 4, 2)["f1"] == pytest.approx(0.625)
    rows = [
        {"doc_id": "a", "tp": 5, "fp": 4, "fn": 2},
        {"doc_id": "a", "tp": 2, "fp": 1, "fn": 3},
        {"doc_id": "b", "tp": 4, "fp": 2, "fn": 1},
    ]
    first = evaluator.bootstrap_micro(rows, iterations=100, seed=42)
    second = evaluator.bootstrap_micro(rows, iterations=100, seed=42)
    assert first == second
    assert first["cluster_count"] == 2
    assert first["intervals"]["f1"]["lower_95"] <= first["intervals"]["f1"]["upper_95"]


def test_material_and_dataset_share_canonical_label():
    evaluator = load_script("evaluate_scirex_batch.py")
    assert evaluator.canonical_label("Material") == "dataset"
    assert evaluator.canonical_label("Dataset") == "dataset"


def test_flexible_metrics_are_one_to_one():
    evaluator = load_script("evaluate_scirex_batch.py")
    predicted = {(0, 5, "dataset"), (0, 7, "dataset")}
    gold = {(0, 7, "dataset")}
    score = evaluator.flexible_metrics(predicted, gold, "MNIST data", "overlap")
    assert score == {"tp": 1, "fp": 1, "fn": 0, "precision": 0.5, "recall": 1.0,
                     "f1": pytest.approx(2 / 3)}


def test_evaluator_reports_missing_coverage_and_split_statistics(monkeypatch, tmp_path):
    evaluator = load_script("evaluate_scirex_batch.py")
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output_dir = tmp_path / "evaluation"
    gold_rows = [
        {
            "example_id": example_id, "doc_id": doc_id, "source_split": "test",
            "length_bucket": "short", "text": "MNIST",
            "entities": [{"text": "MNIST", "label": "Material", "start_char": 0, "end_char": 5}],
        }
        for example_id, doc_id in (("one", "paper-a"), ("two", "paper-b"))
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in gold_rows), encoding="utf-8")
    prediction = {
        "example_id": "one", "doc_id": "paper-a", "source_split": "test",
        "length_bucket": "short", "status": "ok", "processed_char_end": 5,
        "processed_sentences": 1, "predictions": [
            {"text": "MNIST", "type": "Dataset", "start": 0, "end": 5}
        ],
        "elapsed_seconds": 1.0, "invalid_response_count": 0, "llm_calls": 1,
    }
    predictions.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv",
        ["evaluate_scirex_batch.py", "--manifest", str(manifest),
         "--predictions", str(predictions), "--output-dir", str(output_dir)],
    )

    evaluator.main()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["expected_examples"] == 2
    assert summary["completed_examples"] == 1
    assert summary["failed_examples"] == 1
    assert summary["coverage_rate"] == 0.5
    assert summary["missing_prediction_ids"] == ["two"]
    assert summary["groups"][0]["group_type"] in {"bucket", "label", "split"}
    assert summary["split_confidence_intervals"]["test"]["cluster_count"] == 1
