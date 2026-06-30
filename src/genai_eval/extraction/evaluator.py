"""Evaluation helpers for extraction predictions."""

from __future__ import annotations

from collections import defaultdict


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def strict_key(span: dict) -> tuple:
    return (
        normalize_text(span.get("text", "")),
        span.get("start"),
        span.get("end"),
    )


def lenient_key(span: dict) -> tuple:
    return normalize_text(span.get("text", ""))


def compute_prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def match_counts(gold_spans: list[dict], pred_spans: list[dict], key_fn) -> tuple[int, int, int]:
    gold_keys = [key_fn(span) for span in gold_spans]
    pred_keys = [key_fn(span) for span in pred_spans]
    remaining_gold = list(gold_keys)
    tp = 0
    for pred_key in pred_keys:
        if pred_key in remaining_gold:
            remaining_gold.remove(pred_key)
            tp += 1
    fp = len(pred_keys) - tp
    fn = len(gold_keys) - tp
    return tp, fp, fn


def compute_entity_metrics(records: list[dict], key_fn) -> dict:
    total_tp = total_fp = total_fn = 0
    for record in records:
        gold_spans = record.get("gold", {}).get("spans", [])
        pred_spans = record.get("prediction", {}).get("entities", [])
        tp, fp, fn = match_counts(gold_spans, pred_spans, key_fn)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    return compute_prf(total_tp, total_fp, total_fn)


def compute_json_valid_rate(records: list[dict]) -> float:
    applicable = [record for record in records if not record.get("json_validity", {}).get("not_applicable", False)]
    if not applicable:
        return 0.0
    valid = sum(1 for record in applicable if record.get("json_validity", {}).get("all_json_valid", False))
    return round(valid / len(applicable), 4)


def compute_invalid_json_rate(records: list[dict]) -> float:
    applicable = [record for record in records if not record.get("json_validity", {}).get("not_applicable", False)]
    if not applicable:
        return 0.0
    invalid = sum(1 for record in applicable if not record.get("json_validity", {}).get("all_json_valid", False))
    return round(invalid / len(applicable), 4)


def compute_invalid_label_rate(records: list[dict]) -> float:
    if not records:
        return 0.0
    invalid = sum(1 for record in records if record.get("validation", {}).get("has_invalid_labels", False))
    return round(invalid / len(records), 4)


def build_per_type_rows(records: list[dict]) -> list[dict]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for record in records:
        gold_spans = record.get("gold", {}).get("spans", [])
        pred_spans = record.get("prediction", {}).get("entities", [])
        gold_by_type = defaultdict(list)
        pred_by_type = defaultdict(list)
        for span in gold_spans:
            gold_by_type[span.get("type", "")].append(span)
        for span in pred_spans:
            pred_by_type[span.get("type", "")].append(span)
        for entity_type in sorted(set(gold_by_type) | set(pred_by_type)):
            tp, fp, fn = match_counts(gold_by_type[entity_type], pred_by_type[entity_type], strict_key)
            counts[entity_type]["tp"] += tp
            counts[entity_type]["fp"] += fp
            counts[entity_type]["fn"] += fn

    rows = []
    for entity_type, summary in sorted(counts.items()):
        metrics = compute_prf(summary["tp"], summary["fp"], summary["fn"])
        rows.append(
            {
                "entity_type": entity_type,
                "support": summary["tp"] + summary["fn"],
                "tp": summary["tp"],
                "fp": summary["fp"],
                "fn": summary["fn"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )
    return rows


def evaluate_predictions(records: list[dict]) -> dict:
    strict_metrics = compute_entity_metrics(records, strict_key)
    lenient_metrics = compute_entity_metrics(records, lenient_key)
    method = records[0].get("method", "unknown") if records else "unknown"
    model = records[0].get("model", "") if records else ""
    return {
        "method": method,
        "model": model,
        "n_examples": len(records),
        "strict_precision": strict_metrics["precision"],
        "strict_recall": strict_metrics["recall"],
        "strict_f1": strict_metrics["f1"],
        "lenient_precision": lenient_metrics["precision"],
        "lenient_recall": lenient_metrics["recall"],
        "lenient_f1": lenient_metrics["f1"],
        "json_valid_rate": compute_json_valid_rate(records),
        "invalid_json_rate": compute_invalid_json_rate(records),
        "invalid_label_rate": compute_invalid_label_rate(records),
    }


def build_error_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        gold_spans = record.get("gold", {}).get("spans", [])
        pred_spans = record.get("prediction", {}).get("entities", [])
        strict_tp, strict_fp, strict_fn = match_counts(gold_spans, pred_spans, strict_key)
        strict_metrics = compute_prf(strict_tp, strict_fp, strict_fn)
        lenient_tp, lenient_fp, lenient_fn = match_counts(gold_spans, pred_spans, lenient_key)
        lenient_metrics = compute_prf(lenient_tp, lenient_fp, lenient_fn)
        rows.append(
            {
                "id": record.get("id"),
                "text": record.get("text", ""),
                "gold_spans": gold_spans,
                "pred_spans": pred_spans,
                "strict_precision": strict_metrics["precision"],
                "strict_recall": strict_metrics["recall"],
                "strict_f1": strict_metrics["f1"],
                "lenient_precision": lenient_metrics["precision"],
                "lenient_recall": lenient_metrics["recall"],
                "lenient_f1": lenient_metrics["f1"],
                "json_valid": record.get("json_validity", {}).get("all_json_valid"),
                "invalid_entities": record.get("validation", {}).get("invalid_entities", []),
            }
        )
    return rows
