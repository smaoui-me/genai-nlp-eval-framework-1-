"""Validate normalized SciREX documents, benchmark, splits, and fixture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from genai_eval.annotation.validation import validate_processed_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/annotation/scirex_preprocessing.yaml"))
    parser.add_argument("--fixture-dir", type=Path, default=Path("data/annotation/fixtures/scirex"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = validate_processed_dataset(args.processed_dir, config, args.fixture_dir)
    print(
        f"SciREX validation passed: {result['documents']} documents, "
        f"{result['examples']} examples, {result['unique_doc_ids']} unique benchmark doc_id values."
    )


if __name__ == "__main__":
    main()
