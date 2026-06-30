"""Build a location-only FewNERD extraction dataset for method comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from genai_eval.extraction.labels import COARSE_LABELS
from genai_eval.extraction.utils import build_location_spans, parse_int_array, parse_token_array


LOCATION_LABEL = "location"


def find_location_tag_id() -> int:
    for tag_id, label_name in COARSE_LABELS.items():
        if label_name == LOCATION_LABEL:
            return tag_id
    raise ValueError("Could not find coarse FewNERD location tag id")


def process_dataframe(df: pd.DataFrame, max_examples: int) -> pd.DataFrame:
    location_tag_id = find_location_tag_id()
    rows: list[dict] = []
    for _, row in df.iterrows():
        tokens = parse_token_array(row["tokens"])
        coarse_tags = parse_int_array(row["ner_tags"])
        if len(tokens) != len(coarse_tags):
            continue
        gold_entities = build_location_spans(tokens, coarse_tags, location_tag_id)
        if not gold_entities:
            continue
        rows.append(
            {
                "id": row["id"],
                "tokens": json.dumps(tokens, ensure_ascii=False),
                "sentence": " ".join(tokens),
                "gold_entities": json.dumps(gold_entities, ensure_ascii=False),
            }
        )
        if len(rows) >= max_examples:
            break
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build location-only FewNERD eval dataset.")
    parser.add_argument("--input", default="data/extraction/raw/intra/test-00000-of-00001.csv")
    parser.add_argument("--output", default="data/extraction/processed/fewnerd_location_test_500.csv")
    parser.add_argument("--max-examples", type=int, default=500)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    processed = process_dataframe(df, max_examples=args.max_examples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    print(f"Saved {len(processed)} rows to {output_path}")


if __name__ == "__main__":
    main()
