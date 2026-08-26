"""Fail-fast, network-free preflight for the SciREX 100/1,000 study."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "genai-nlp-annotation-tool"
sys.path.insert(0, str(APP_DIR))

from utils.tokenizer import split_sentences  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runner_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return value.get("runner", value)


def inspect_manifest(path: Path) -> dict:
    rows = read_jsonl(path)
    ids = [row["example_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate example_id in {path}")
    doc_splits = defaultdict(set)
    bad_offsets = []
    for row in rows:
        doc_splits[row["doc_id"]].add(row["source_split"])
        text = row["text"]
        for entity in row["entities"]:
            start, end = int(entity["start_char"]), int(entity["end_char"])
            if text[start:end] != entity["text"]:
                bad_offsets.append((row["example_id"], start, end, entity["text"][:50]))
    if bad_offsets:
        raise ValueError(f"Gold offset assertions failed; first errors: {bad_offsets[:3]}")
    crossed = [doc_id for doc_id, splits in doc_splits.items() if len(splits) != 1]
    if crossed:
        raise ValueError(f"Source papers cross splits: {crossed[:3]}")
    return {
        "path": path.as_posix(),
        "sha256": sha256(path),
        "examples": len(rows),
        "unique_papers": len(doc_splits),
        "splits": dict(sorted(Counter(row["source_split"] for row in rows).items())),
        "length_buckets": dict(sorted(Counter(row["length_bucket"] for row in rows).items())),
        "gold_entities": sum(len(row["entities"]) for row in rows),
        "source_sentences": sum(int(row["sentence_count"]) for row in rows),
        "app_calls": sum(len(split_sentences(row["text"])) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operational-config", type=Path,
        default=Path("configs/annotation/scirex_operational_100.yaml"),
    )
    parser.add_argument(
        "--full-config", type=Path,
        default=Path("configs/annotation/scirex_full_1000.yaml"),
    )
    args = parser.parse_args()

    configs = {
        "operational": load_runner_config(args.operational_config),
        "full": load_runner_config(args.full_config),
    }
    reports = {}
    warnings = []
    for name, config in configs.items():
        if int(config.get("max_sentences", -1)) != 0:
            raise ValueError(f"{name}: max_sentences must be 0 for complete-window evaluation")
        report = inspect_manifest(Path(config["input"]))
        if int(config.get("max_calls", 0)) < report["app_calls"]:
            raise ValueError(
                f"{name}: max_calls={config.get('max_calls')} is below planned calls={report['app_calls']}"
            )
        if not float(config.get("max_cost_usd", 0)):
            warnings.append(f"{name}: monetary spend ceiling is disabled")
        if not (
            float(config.get("input_cost_per_million", 0))
            or float(config.get("output_cost_per_million", 0))
        ):
            warnings.append(f"{name}: provider token prices are not configured")
        reports[name] = report

    operational_rows = read_jsonl(Path(configs["operational"]["input"]))
    if len(operational_rows) != 100 or len({row["doc_id"] for row in operational_rows}) != 100:
        raise ValueError("Operational gate must contain 100 document-disjoint windows")
    if {row["source_split"] for row in operational_rows} != {"train"}:
        raise ValueError("Operational gate must use training papers only; held-out leakage detected")
    if Counter(row["length_bucket"] for row in operational_rows) != Counter(
        {"short": 25, "medium": 25, "long": 25, "very_long": 25}
    ):
        raise ValueError("Operational gate must contain 25 windows per length bucket")

    full_rows = read_jsonl(Path(configs["full"]["input"]))
    if len(full_rows) != 1000:
        raise ValueError(f"Full manifest must contain 1,000 windows, found {len(full_rows)}")
    if not {"train", "dev", "test"}.issubset({row["source_split"] for row in full_rows}):
        raise ValueError("Full manifest does not contain all official source splits")

    print(json.dumps({"ready": True, "manifests": reports, "warnings": warnings}, indent=2))


if __name__ == "__main__":
    main()
