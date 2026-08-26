"""Prepare normalized SciREX documents, benchmark windows, and fixtures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genai_eval.annotation.chunking import BUCKET_NAMES, build_benchmark  # noqa: E402
from genai_eval.annotation.scirex import load_and_normalize, observed_labels  # noqa: E402
from genai_eval.annotation.validation import validate_processed_dataset  # noqa: E402


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def summary(values: list[int]) -> dict:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": min(values), "max": max(values),
        "mean": round(statistics.mean(values), 3),
        "median": statistics.median(values),
    }


def overlap_statistics(examples: list[dict]) -> dict:
    grouped = defaultdict(list)
    for item in examples:
        grouped[item["doc_id"]].append((item["source_sentence_start"], item["source_sentence_end_exclusive"]))
    ratios = []
    for windows in grouped.values():
        for index, first in enumerate(windows):
            for second in windows[index + 1:]:
                overlap = max(0, min(first[1], second[1]) - max(first[0], second[0]))
                if overlap:
                    ratios.append(overlap / min(first[1] - first[0], second[1] - second[0]))
    return {
        "overlapping_window_pairs": len(ratios),
        "maximum_overlap_ratio": round(max(ratios), 6) if ratios else 0.0,
        "mean_overlap_ratio_when_overlapping": round(statistics.mean(ratios), 6) if ratios else 0.0,
    }


def build_statistics(documents: list[dict], examples: list[dict], labels: Counter, candidates: dict) -> dict:
    bucket_stats = {}
    for bucket in BUCKET_NAMES:
        selected = [item for item in examples if item["length_bucket"] == bucket]
        bucket_stats[bucket] = {
            "count": len(selected),
            "sentences": summary([item["sentence_count"] for item in selected]),
            "characters": summary([len(item["text"]) for item in selected]),
            "entity_count": sum(len(item["entities"]) for item in selected),
            "zero_entity_windows": sum(not item["entities"] for item in selected),
        }
    windows_per_doc = Counter(item["doc_id"] for item in examples)
    return {
        "raw_document_count_by_split": dict(Counter(doc["source_split"] for doc in documents)),
        "total_raw_document_count": len(documents),
        "raw_token_count": sum(doc["token_count"] for doc in documents),
        "raw_sentence_count": sum(doc["sentence_count"] for doc in documents),
        "raw_section_count": sum(doc["section_count"] for doc in documents),
        "raw_entity_count": sum(doc["entity_count"] for doc in documents),
        "observed_entity_labels": dict(sorted(labels.items())),
        "source_document_sentence_length": summary([doc["sentence_count"] for doc in documents]),
        "benchmark_count_by_bucket": dict(Counter(item["length_bucket"] for item in examples)),
        "benchmark_count_by_source_split": dict(Counter(item["source_split"] for item in examples)),
        "unique_source_doc_id_count": len(windows_per_doc),
        "windows_per_source_document": dict(sorted(Counter(windows_per_doc.values()).items())),
        "overlap": overlap_statistics(examples),
        "buckets": bucket_stats,
        "candidate_report": candidates,
    }


def write_fixtures(examples: list[dict], fixture_dir: Path, config: dict) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    per_bucket = int(config.get("fixtures", {}).get("examples_per_bucket", 2))
    selected = []
    for bucket in BUCKET_NAMES:
        bucket_examples = [item for item in examples if item["length_bucket"] == bucket]
        chosen = []
        used_docs = set()
        for item in bucket_examples:
            if item["doc_id"] not in used_docs:
                chosen.append(item)
                used_docs.add(item["doc_id"])
            if len(chosen) == per_bucket:
                break
        selected.extend(chosen)
    write_jsonl(fixture_dir / "annotation_tool_smoke.jsonl", selected)
    expected = {
        "expected_example_count": per_bucket * len(BUCKET_NAMES),
        "expected_buckets": {bucket: per_bucket for bucket in BUCKET_NAMES},
        "required_fields": [
            "example_id", "doc_id", "source_split", "length_bucket", "sentence_count",
            "text", "sentences", "sections", "entities",
        ],
    }
    (fixture_dir / "annotation_tool_smoke_expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_manual_review(examples: list[dict], path: Path, config: dict) -> None:
    per_bucket = int(config.get("fixtures", {}).get("manual_review_examples_per_bucket", 8))
    preview_chars = int(config.get("fixtures", {}).get("preview_characters", 500))
    selected = []
    for bucket in BUCKET_NAMES:
        selected.extend([item for item in examples if item["length_bucket"] == bucket][:per_bucket])
    fields = [
        "example_id", "doc_id", "source_split", "length_bucket", "sentence_count",
        "character_count", "entity_count", "section_count", "text_preview", "first_entities",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in selected:
            writer.writerow({
                "example_id": item["example_id"], "doc_id": item["doc_id"],
                "source_split": item["source_split"], "length_bucket": item["length_bucket"],
                "sentence_count": item["sentence_count"], "character_count": len(item["text"]),
                "entity_count": len(item["entities"]), "section_count": len(item["sections"]),
                "text_preview": item["text"][:preview_chars],
                "first_entities": json.dumps(item["entities"][:5], ensure_ascii=False),
            })


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target-examples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-redistribute", action="store_true")
    parser.add_argument("--fixture-output-dir", type=Path, default=Path("data/annotation/fixtures/scirex"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:
        config["seed"] = args.seed
    if args.target_examples is not None:
        if args.target_examples % 4:
            raise ValueError("--target-examples must be divisible by four")
        config["target_examples"] = args.target_examples
        for bucket in BUCKET_NAMES:
            config["buckets"][bucket]["target"] = args.target_examples // 4

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {args.output_dir}; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and normalizing SciREX train/dev/test documents...")
    documents, input_files = load_and_normalize(args.raw_dir)
    for split in ("train", "dev", "test"):
        print(f"  {split}: {sum(doc['source_split'] == split for doc in documents)} documents")
    labels = observed_labels(documents)
    write_jsonl(args.output_dir / "documents.jsonl", documents)

    print("Building deterministic contiguous same-document benchmark...")
    examples, candidates = build_benchmark(documents, config, args.allow_redistribute)
    write_jsonl(args.output_dir / "examples_1000.jsonl", examples)
    for split in ("train", "dev", "test"):
        write_jsonl(args.output_dir / f"{split}.jsonl", [x for x in examples if x["source_split"] == split])

    label_schema = {
        "dataset": "scirex",
        "labels": [{"name": label, "description": "SciREX entity label"} for label in sorted(labels)],
    }
    (args.output_dir / "label_schema.json").write_text(
        json.dumps(label_schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_fixtures(examples, args.fixture_output_dir, config)
    write_manual_review(examples, args.output_dir / "manual_review_sample.csv", config)
    stats = build_statistics(documents, examples, labels, candidates)
    (args.output_dir / "statistics.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    output_files = sorted(path for path in args.output_dir.iterdir() if path.is_file())
    output_files.extend(sorted(path for path in args.fixture_output_dir.iterdir() if path.is_file()))
    manifest = {
        "dataset": "scirex", "schema_version": config.get("schema_version", "1.0"),
        "created_at": datetime.now(timezone.utc).isoformat(), "preprocessing_seed": config["seed"],
        "configuration": config,
        "inputs": [{
            "split": split, "path": relative_path(path), "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        } for split, path in sorted(input_files.items())],
        "outputs": [{
            "path": relative_path(path), "record_count": (
                sum(1 for _ in path.open(encoding="utf-8")) if path.suffix == ".jsonl" else None
            ), "size_bytes": path.stat().st_size, "sha256": sha256(path),
        } for path in output_files],
        "git_commit": git_commit(),
        "preprocessing_command": (
            "python scripts/annotation/prepare_scirex.py "
            "--raw-dir data/annotation/raw/release_data "
            "--output-dir data/annotation/processed/scirex "
            "--config configs/annotation/scirex_preprocessing.yaml --overwrite"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    validation = validate_processed_dataset(args.output_dir, config, args.fixture_output_dir)
    print(f"Complete: {validation['documents']} documents, {validation['examples']} examples, "
          f"{validation['unique_doc_ids']} unique benchmark documents")
    print(f"Buckets: {stats['benchmark_count_by_bucket']}")
    print(f"Splits: {stats['benchmark_count_by_source_split']}")
    print(f"Labels: {dict(labels)}")


if __name__ == "__main__":
    main()
