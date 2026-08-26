"""Aggregate strict SciREX span-and-label metrics from batch predictions."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]")


def canonical_label(label: str) -> str:
    return "dataset" if str(label).casefold() in {"material", "dataset"} else str(label).casefold()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(entity: dict, gold: bool = False) -> tuple[int, int, str]:
    return (
        int(entity["start_char" if gold else "start"]),
        int(entity["end_char" if gold else "end"]),
        canonical_label(entity["label" if gold else "type"]),
    )


def metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _token_span(span: tuple[int, int, str], token_offsets: list[tuple[int, int]]) -> tuple[int, int]:
    start, end, _ = span
    touched = [i for i, (a, b) in enumerate(token_offsets) if max(a, start) < min(b, end)]
    return (touched[0], touched[-1]) if touched else (-10_000, -10_000)


def flexible_metrics(predicted: set, reference: set, text: str, mode: str) -> dict:
    """One-to-one same-label matching for tolerant-boundary or overlap scoring."""
    token_offsets = [(match.start(), match.end()) for match in TOKEN_RE.finditer(text)]
    token_spans = {
        span: _token_span(span, token_offsets) for span in predicted | reference
    } if mode == "boundary_tolerant" else {}
    predictions_by_label = defaultdict(list)
    gold_by_label = defaultdict(list)
    for span in predicted:
        predictions_by_label[span[2]].append(span)
    for span in reference:
        gold_by_label[span[2]].append(span)
    candidates = []
    for label, label_predictions in predictions_by_label.items():
        for prediction in label_predictions:
          for gold in gold_by_label.get(label, []):
            intersection = max(0, min(prediction[1], gold[1]) - max(prediction[0], gold[0]))
            if mode == "overlap" and intersection:
                union = max(prediction[1], gold[1]) - min(prediction[0], gold[0])
                candidates.append((intersection / union, prediction, gold))
            elif mode == "boundary_tolerant":
                p_start, p_end = token_spans[prediction]
                g_start, g_end = token_spans[gold]
                if abs(p_start - g_start) <= 1 and abs(p_end - g_end) <= 2:
                    candidates.append((-(abs(p_start - g_start) + abs(p_end - g_end)), prediction, gold))
    used_predictions, used_gold = set(), set()
    for _, prediction, gold in sorted(candidates, reverse=True):
        if prediction not in used_predictions and gold not in used_gold:
            used_predictions.add(prediction); used_gold.add(gold)
    return metrics(len(used_predictions), len(predicted) - len(used_predictions), len(reference) - len(used_gold))


def bootstrap_micro(per_example: list[dict], iterations: int = 2000, seed: int = 42) -> dict:
    """Clustered percentile intervals: resample papers, retaining all their windows."""
    if not per_example:
        return {}
    rng = random.Random(seed)
    clusters = defaultdict(list)
    for row in per_example:
        clusters[row["doc_id"]].append(row)
    cluster_values = list(clusters.values())
    samples = {name: [] for name in ("precision", "recall", "f1")}
    for _ in range(iterations):
        drawn = [row for _ in cluster_values for row in rng.choice(cluster_values)]
        score = metrics(
            sum(row["tp"] for row in drawn),
            sum(row["fp"] for row in drawn),
            sum(row["fn"] for row in drawn),
        )
        for name in samples:
            samples[name].append(score[name])
    intervals = {}
    for name, values in samples.items():
        values.sort()
        intervals[name] = {
            "lower_95": values[int(0.025 * iterations)],
            "upper_95": values[min(iterations - 1, int(0.975 * iterations))],
        }
    return {"method": "doc_id-clustered percentile bootstrap", "cluster_count": len(cluster_values),
            "iterations": iterations, "seed": seed,
            "intervals": intervals}


def macro_document_f1(per_example: list[dict]) -> float:
    documents = defaultdict(lambda: defaultdict(int))
    for row in per_example:
        for name in ("tp", "fp", "fn"):
            documents[row["doc_id"]][name] += row[name]
    values = [metrics(value["tp"], value["fp"], value["fn"])["f1"] for value in documents.values()]
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("eval/corpora/scirex_dev_pilot_20.jsonl"))
    parser.add_argument("--predictions", type=Path, default=Path("results/annotation/scirex_dev_pilot_20/predictions.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/annotation/scirex_dev_pilot_20/evaluation"))
    args = parser.parse_args()
    gold_by_id = {row["example_id"]: row for row in read_jsonl(args.manifest)}
    latest = {}
    for row in read_jsonl(args.predictions):
        latest[row["example_id"]] = row

    totals = defaultdict(int)
    flexible_totals = {mode: defaultdict(int) for mode in ("boundary_tolerant", "overlap")}
    groups = defaultdict(lambda: defaultdict(int))
    flexible_groups = defaultdict(lambda: defaultdict(int))
    per_example, errors, false_positives, false_negatives = [], [], [], []
    missing_prediction_ids = sorted(set(gold_by_id) - set(latest))
    unknown_prediction_ids = sorted(set(latest) - set(gold_by_id))
    for example_id, gold in gold_by_id.items():
        run = latest.get(example_id)
        if run is None:
            errors.append({"example_id": example_id, "status": "missing", "error": "No prediction row"})
            continue
        if run.get("status") != "ok":
            errors.append(run)
            continue
        cutoff = int(run["processed_char_end"])
        predicted = {key(x) for x in run["predictions"] if int(x["end"]) <= cutoff}
        reference = {key(x, gold=True) for x in gold["entities"] if int(x["end_char"]) <= cutoff}
        tp, fp, fn = len(predicted & reference), len(predicted - reference), len(reference - predicted)
        score = metrics(tp, fp, fn)
        flexible = {
            mode: flexible_metrics(predicted, reference, gold["text"], mode)
            for mode in flexible_totals
        }
        per_example.append({
            "example_id": example_id, "doc_id": run["doc_id"], "split": run["source_split"],
            "bucket": run["length_bucket"],
            "processed_sentences": run["processed_sentences"], "predicted": len(predicted),
            "gold": len(reference), **score, "elapsed_seconds": run["elapsed_seconds"],
            "invalid_response_count": run["invalid_response_count"],
            "boundary_tolerant_f1": flexible["boundary_tolerant"]["f1"],
            "overlap_f1": flexible["overlap"]["f1"],
        })
        for name in ("tp", "fp", "fn"):
            totals[name] += score[name]
            groups[("bucket", run["length_bucket"])][name] += score[name]
            groups[("split", run["source_split"])][name] += score[name]
        for mode, mode_score in flexible.items():
            for name in ("tp", "fp", "fn"):
                flexible_totals[mode][name] += mode_score[name]
                flexible_groups[(mode, "bucket", run["length_bucket"])][name] += mode_score[name]
                flexible_groups[(mode, "split", run["source_split"])][name] += mode_score[name]
        for label in {x[2] for x in predicted | reference}:
            p = {x for x in predicted if x[2] == label}
            g = {x for x in reference if x[2] == label}
            groups[("label", label)]["tp"] += len(p & g)
            groups[("label", label)]["fp"] += len(p - g)
            groups[("label", label)]["fn"] += len(g - p)
        for error_type, spans in (("false_positive", predicted - reference), ("false_negative", reference - predicted)):
            target = false_positives if error_type == "false_positive" else false_negatives
            for start, end, label in sorted(spans):
                target.append({
                    "example_id": example_id, "doc_id": run["doc_id"],
                    "bucket": run["length_bucket"], "label": label,
                    "start_char": start, "end_char": end,
                    "text": gold["text"][start:end],
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {
        split: [row for row in per_example if row["split"] == split]
        for split in sorted({row["split"] for row in per_example})
    }
    summary = {
        "evaluation": "strict exact character span + case-insensitive label",
        "expected_examples": len(gold_by_id),
        "completed_examples": len(per_example), "failed_examples": len(errors),
        "coverage_rate": len(per_example) / len(gold_by_id) if gold_by_id else 0.0,
        "missing_prediction_ids": missing_prediction_ids,
        "unknown_prediction_ids": unknown_prediction_ids,
        "micro": metrics(totals["tp"], totals["fp"], totals["fn"]),
        "boundary_tolerant_micro": metrics(**flexible_totals["boundary_tolerant"]),
        "overlap_micro": metrics(**flexible_totals["overlap"]),
        "macro_document_f1": macro_document_f1(per_example),
        "confidence_intervals": bootstrap_micro(per_example),
        "split_confidence_intervals": {
            split: bootstrap_micro(rows) for split, rows in split_rows.items()
        },
        "split_macro_document_f1": {
            split: macro_document_f1(rows) for split, rows in split_rows.items()
        },
        "total_llm_calls": sum(row.get("llm_calls", 0) for row in latest.values()),
        "total_elapsed_seconds": sum(row.get("elapsed_seconds", 0) for row in latest.values()),
        "input_tokens": sum(row.get("input_tokens", 0) for row in latest.values()),
        "output_tokens": sum(row.get("output_tokens", 0) for row in latest.values()),
        "total_tokens": sum(row.get("total_tokens", 0) for row in latest.values()),
        "usage_missing_calls": sum(row.get("usage_missing_calls", 0) for row in latest.values()),
        "estimated_cost_usd": round(sum(row.get("estimated_cost_usd", 0) for row in latest.values()), 6),
        "invalid_response_count": sum(row.get("invalid_response_count", 0) for row in latest.values()),
        "groups": [
            {"group_type": kind, "group": value, **metrics(x["tp"], x["fp"], x["fn"])}
            for (kind, value), x in sorted(groups.items())
        ],
        "flexible_groups": [
            {"metric": mode, "group_type": kind, "group": value,
             **metrics(x["tp"], x["fp"], x["fn"])}
            for (mode, kind, value), x in sorted(flexible_groups.items())
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if per_example:
        with (args.output_dir / "per_example.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_example[0]))
            writer.writeheader(); writer.writerows(per_example)
    for filename, rows in (("false_positives.csv", false_positives), ("false_negatives.csv", false_negatives)):
        with (args.output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            fieldnames = ["example_id", "doc_id", "bucket", "label", "start_char", "end_char", "text"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "failed_examples.json").write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
