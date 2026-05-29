"""
extraction_evaluator.py

Shared evaluation functions for all ticket extraction methods.
Computes accuracy and F1 for type and queue (single-label),
micro and row-level precision/recall/F1 for tags (multi-label),
and an evidence validity rate.

All comparisons use normalized strings (lowercase, stripped, collapsed whitespace).
"""

import re
from collections import defaultdict
from typing import Optional

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Normalize a label or evidence string for comparison.

    Lowercases, strips leading/trailing whitespace, and collapses
    internal repeated whitespace to a single space.

    Args:
        text: Input string.

    Returns:
        Normalized string.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# Single-label metrics (type, queue)
# ---------------------------------------------------------------------------

def compute_single_label_metrics(gold_list: list, pred_list: list) -> dict:
    """Compute accuracy and macro-F1 for a single-label field.

    Args:
        gold_list: List of gold label strings.
        pred_list: List of predicted label strings.

    Returns:
        Dict with keys "accuracy" and "macro_f1".
    """
    gold_norm = [normalize(g) for g in gold_list]
    pred_norm = [normalize(p) for p in pred_list]

    accuracy = accuracy_score(gold_norm, pred_norm)
    macro_f1 = f1_score(gold_norm, pred_norm, average="macro", zero_division=0)
    return {"accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4)}


# ---------------------------------------------------------------------------
# Multi-label tag metrics
# ---------------------------------------------------------------------------

def tag_row_metrics(gold_tags: list, pred_tags: list) -> dict:
    """Compute precision, recall, and F1 for a single ticket's tags.

    Args:
        gold_tags: List of gold tag strings.
        pred_tags: List of predicted tag strings.

    Returns:
        Dict with keys "precision", "recall", "f1", "tp", "fp", "fn".
    """
    gold_set = {normalize(t) for t in gold_tags}
    pred_set = {normalize(t) for t in pred_tags}

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_tag_metrics(gold_tag_lists: list, pred_tag_lists: list) -> dict:
    """Compute micro and macro (row-averaged) tag metrics over the full dataset.

    Args:
        gold_tag_lists: List of gold tag lists (one per ticket).
        pred_tag_lists: List of predicted tag lists (one per ticket).

    Returns:
        Dict with micro and row-level precision, recall, F1.
    """
    total_tp = total_fp = total_fn = 0
    row_precisions, row_recalls, row_f1s = [], [], []

    for gold, pred in zip(gold_tag_lists, pred_tag_lists):
        row = tag_row_metrics(gold, pred)
        total_tp += row["tp"]
        total_fp += row["fp"]
        total_fn += row["fn"]
        row_precisions.append(row["precision"])
        row_recalls.append(row["recall"])
        row_f1s.append(row["f1"])

    micro_precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    )
    micro_recall = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    n = len(row_f1s)
    row_precision_avg = sum(row_precisions) / n if n else 0.0
    row_recall_avg = sum(row_recalls) / n if n else 0.0
    row_f1_avg = sum(row_f1s) / n if n else 0.0

    return {
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "row_precision": round(row_precision_avg, 4),
        "row_recall": round(row_recall_avg, 4),
        "row_f1": round(row_f1_avg, 4),
    }


# ---------------------------------------------------------------------------
# Evidence validity
# ---------------------------------------------------------------------------

def is_evidence_valid(evidence: str, text: str) -> bool:
    """Check whether a normalized evidence string appears in the normalized text.

    Args:
        evidence: Evidence snippet string.
        text: Full ticket text.

    Returns:
        True if evidence is non-empty and is a substring of the normalized text.
    """
    norm_ev = normalize(evidence)
    norm_text = normalize(text)
    return bool(norm_ev) and norm_ev in norm_text


def compute_evidence_valid_rate(records: list) -> float:
    """Compute the fraction of evidence strings that appear in ticket text.

    Checks type evidence, queue evidence, and all tag evidences.

    Args:
        records: List of prediction records, each containing "text" and "prediction".

    Returns:
        evidence_valid_rate as a float in [0, 1].
    """
    total = 0
    valid = 0

    for record in records:
        text = record.get("text", "")
        prediction = record.get("prediction", {})

        for field in ("type", "queue"):
            evidence = prediction.get(field, {}).get("evidence", "")
            if evidence:
                total += 1
                if is_evidence_valid(evidence, text):
                    valid += 1

        for tag in prediction.get("tags", []):
            evidence = tag.get("evidence", "")
            if evidence:
                total += 1
                if is_evidence_valid(evidence, text):
                    valid += 1

    return round(valid / total, 4) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_predictions(records: list) -> dict:
    """Evaluate a list of prediction records and return aggregated metrics.

    Args:
        records: List of dicts each containing "gold" and "prediction" keys.

    Returns:
        Dict with all scalar metric values.
    """
    gold_types, pred_types = [], []
    gold_queues, pred_queues = [], []
    gold_tags_all, pred_tags_all = [], []

    for record in records:
        gold = record.get("gold", {})
        pred = record.get("prediction", {})

        gold_types.append(gold.get("type", ""))
        pred_types.append(pred.get("type", {}).get("label", ""))

        gold_queues.append(gold.get("queue", ""))
        pred_queues.append(pred.get("queue", {}).get("label", ""))

        gold_tags_all.append(gold.get("tags", []))
        pred_tags_all.append([t["label"] for t in pred.get("tags", [])])

    type_metrics = compute_single_label_metrics(gold_types, pred_types)
    queue_metrics = compute_single_label_metrics(gold_queues, pred_queues)
    tag_metrics = compute_tag_metrics(gold_tags_all, pred_tags_all)
    evidence_rate = compute_evidence_valid_rate(records)

    method = records[0].get("method", "unknown") if records else "unknown"

    return {
        "method": method,
        "n_examples": len(records),
        "type_accuracy": type_metrics["accuracy"],
        "type_macro_f1": type_metrics["macro_f1"],
        "queue_accuracy": queue_metrics["accuracy"],
        "queue_macro_f1": queue_metrics["macro_f1"],
        "tag_micro_precision": tag_metrics["micro_precision"],
        "tag_micro_recall": tag_metrics["micro_recall"],
        "tag_micro_f1": tag_metrics["micro_f1"],
        "tag_row_precision": tag_metrics["row_precision"],
        "tag_row_recall": tag_metrics["row_recall"],
        "tag_row_f1": tag_metrics["row_f1"],
        "evidence_valid_rate": evidence_rate,
    }


def build_error_rows(records: list) -> list:
    """Build a list of error rows for tickets with wrong type, queue, or tag F1 < 1.

    Args:
        records: List of full prediction records.

    Returns:
        List of dicts suitable for writing to an errors CSV.
    """
    error_rows = []

    for record in records:
        gold = record.get("gold", {})
        pred = record.get("prediction", {})
        text = record.get("text", "")

        gold_type = gold.get("type", "")
        pred_type = pred.get("type", {}).get("label", "")
        type_correct = normalize(gold_type) == normalize(pred_type)

        gold_queue = gold.get("queue", "")
        pred_queue = pred.get("queue", {}).get("label", "")
        queue_correct = normalize(gold_queue) == normalize(pred_queue)

        gold_tags = gold.get("tags", [])
        pred_tags = [t["label"] for t in pred.get("tags", [])]
        tag_row = tag_row_metrics(gold_tags, pred_tags)

        # Evidence rate for this single record
        row_evidence_rate = compute_evidence_valid_rate([record])

        is_error = (not type_correct) or (not queue_correct) or (tag_row["f1"] < 1.0)
        if is_error:
            error_rows.append(
                {
                    "ticket_id": record.get("ticket_id"),
                    "text": text,
                    "gold_type": gold_type,
                    "pred_type": pred_type,
                    "type_correct": type_correct,
                    "gold_queue": gold_queue,
                    "pred_queue": pred_queue,
                    "queue_correct": queue_correct,
                    "gold_tags": gold_tags,
                    "pred_tags": pred_tags,
                    "tag_precision": tag_row["precision"],
                    "tag_recall": tag_row["recall"],
                    "tag_f1": tag_row["f1"],
                    "evidence_valid_rate": row_evidence_rate,
                }
            )

    return error_rows
