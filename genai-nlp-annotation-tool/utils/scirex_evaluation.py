"""Compare immutable model predictions in an export with hidden SciREX gold."""

from __future__ import annotations

from utils.benchmark_data import find_scirex_example
from utils.prompt_builder import canonical_label
from utils.tokenizer import tokenize


def _key(entity: dict) -> tuple[int, int, str]:
    return int(entity["start"]), int(entity["end"]), canonical_label(entity["type"])


def _gold_key(entity: dict) -> tuple[int, int, str]:
    return int(entity["start_char"]), int(entity["end_char"]), canonical_label(entity["label"])


def _scores(predicted: set, gold: set) -> dict:
    tp, fp, fn = len(predicted & gold), len(predicted - gold), len(gold - predicted)
    return _scores_from_counts(tp, fp, fn)


def _scores_from_counts(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
    }


def _flexible_scores(predicted: set, gold: set, text_value: str, mode: str) -> dict:
    tokens = tokenize(text_value)
    offsets = [(token.start, token.end) for token in tokens]

    def token_span(span):
        touched = [i for i, (start, end) in enumerate(offsets) if max(start, span[0]) < min(end, span[1])]
        return (touched[0], touched[-1]) if touched else (-10_000, -10_000)

    token_spans = {span: token_span(span) for span in predicted | gold} if mode == "boundary_tolerant" else {}
    candidates = []
    for prediction in predicted:
        for reference in gold:
            if prediction[2] != reference[2]:
                continue
            intersection = max(0, min(prediction[1], reference[1]) - max(prediction[0], reference[0]))
            if mode == "overlap" and intersection:
                union = max(prediction[1], reference[1]) - min(prediction[0], reference[0])
                candidates.append((intersection / union, prediction, reference))
            elif mode == "boundary_tolerant":
                p_start, p_end = token_spans[prediction]
                g_start, g_end = token_spans[reference]
                if abs(p_start - g_start) <= 1 and abs(p_end - g_end) <= 2:
                    candidates.append((-(abs(p_start - g_start) + abs(p_end - g_end)), prediction, reference))
    used_predictions, used_gold = set(), set()
    for _, prediction, reference in sorted(candidates, reverse=True):
        if prediction not in used_predictions and reference not in used_gold:
            used_predictions.add(prediction); used_gold.add(reference)
    return _scores_from_counts(
        len(used_predictions),
        len(predicted) - len(used_predictions),
        len(gold) - len(used_gold),
    )


def evaluate_scirex_export(export: dict) -> dict:
    source = export.get("source") or {}
    if source.get("source_dataset") != "scirex" or not source.get("example_id"):
        return {"available": False, "reason": "Export is not linked to a SciREX example."}
    example = find_scirex_example(source["example_id"])
    if example is None:
        return {"available": False, "reason": "Matching SciREX benchmark example was not found."}

    predictions = export.get("model_predictions")
    legacy = predictions is None
    if legacy:
        log = [item for item in export.get("review_log", []) if item.get("source") == "model"]
        if any(item.get("status") == "edited" for item in log):
            return {
                "available": False,
                "reason": "Legacy export contains relabels and did not preserve original model labels.",
            }
        predictions = log

    cutoff = int((export.get("uncertainty") or {}).get("processed_char_end") or 0)
    scope = "exact"
    if cutoff <= 0:
        processed_sentences = int((export.get("uncertainty") or {}).get("n_sentences") or 0)
        if processed_sentences and example.get("sentences"):
            cutoff = example["sentences"][min(processed_sentences, len(example["sentences"])) - 1]["end_char"]
            scope = "inferred-from-source-sentences"
        else:
            cutoff = len(example["text"])
            scope = "full-example-fallback"

    predictions = [item for item in predictions if int(item["end"]) <= cutoff]
    gold_entities = [item for item in example["entities"] if int(item["end_char"]) <= cutoff]
    predicted_keys = {_key(item) for item in predictions}
    gold_keys = {_gold_key(item) for item in gold_entities}
    overall = _scores(predicted_keys, gold_keys)
    boundary_tolerant = _flexible_scores(predicted_keys, gold_keys, example["text"], "boundary_tolerant")
    overlap = _flexible_scores(predicted_keys, gold_keys, example["text"], "overlap")

    labels = sorted({key[2] for key in predicted_keys | gold_keys})
    per_label = {
        label: _scores(
            {key for key in predicted_keys if key[2] == label},
            {key for key in gold_keys if key[2] == label},
        ) for label in labels
    }
    return {
        "available": True, "legacy_prediction_reconstruction": legacy,
        "scope": scope, "processed_char_end": cutoff,
        "predicted_entities": len(predicted_keys), "gold_entities": len(gold_keys),
        "strict": overall, "boundary_tolerant": boundary_tolerant, "overlap": overlap,
        "per_label": per_label,
        "false_positives": [list(item) for item in sorted(predicted_keys - gold_keys)],
        "false_negatives": [list(item) for item in sorted(gold_keys - predicted_keys)],
    }
