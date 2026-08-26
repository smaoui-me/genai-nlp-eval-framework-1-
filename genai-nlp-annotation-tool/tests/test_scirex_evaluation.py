"""Focused checks for strict prediction-versus-SciREX-gold evaluation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.benchmark_data import load_scirex_examples
from utils.scirex_evaluation import evaluate_scirex_export


example = next(item for item in load_scirex_examples() if len(item["entities"]) >= 2)
predictions = [
    {
        "text": entity["text"],
        "type": entity["label"],
        "start": entity["start_char"],
        "end": entity["end_char"],
    }
    for entity in example["entities"][:-1]
]
predictions.append({"text": example["text"][0:1], "type": "NotAGoldLabel", "start": 0, "end": 1})

export = {
    "source": {"source_dataset": "scirex", "example_id": example["example_id"]},
    "uncertainty": {"processed_char_end": len(example["text"])},
    "model_predictions": predictions,
    # Human output is intentionally unrelated: it must never leak into model scoring.
    "gold_entities": [],
}
result = evaluate_scirex_export(export)
assert result["available"]
assert result["strict"]["tp"] == len(example["entities"]) - 1
assert result["strict"]["fp"] == 1
assert result["strict"]["fn"] == 1
assert result["scope"] == "exact"

legacy = {
    "source": export["source"],
    "uncertainty": {"n_sentences": 1},
    "review_log": [{"source": "model", "status": "edited", "type": "Method", "start": 0, "end": 1}],
}
legacy_result = evaluate_scirex_export(legacy)
assert not legacy_result["available"]
assert "original model labels" in legacy_result["reason"]

print("SciREX evaluation checks passed.")
