"""Build a location-only FewNERD extraction evaluation dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from genai_eval.extraction.labels import COARSE_LABELS
from genai_eval.extraction.utils import build_location_spans, parse_int_array, parse_token_array, sentence_from_tokens, to_json_string

LOCATION_TAG_ID = next(tag_id for tag_id, label in COARSE_LABELS.items() if label == "location")


def process_dataframe(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    processed_rows = []
    for _, row in df.iterrows():
        tokens = parse_token_array(row["tokens"])
        coarse_tags = parse_int_array(row["ner_tags"])
        if len(tokens) != len(coarse_tags):
            raise ValueError(f"Token/tag length mismatch for id={row['id']}")
        gold_entities = build_location_spans(tokens, coarse_tags, LOCATION_TAG_ID)
        if not gold_entities:
            continue
        processed_rows.append(
            {
                "id": row["id"],
                "sentence": sentence_from_tokens(tokens),
                "tokens": to_json_string(tokens),
                "gold_entities": to_json_string(gold_entities),
            }
        )
        if len(processed_rows) >= limit:
            break
    return pd.DataFrame(processed_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FewNERD location extraction eval data.")
    parser.add_argument("--input", default="data/extraction/raw/intra/test-00000-of-00001.csv")
    parser.add_argument("--output", default="data/extraction/processed/fewnerd_location_test_1000.csv")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    processed = process_dataframe(df, limit=args.limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed.to_csv(output_path, index=False)

    print(f"Saved {len(processed)} rows to {output_path}")


if __name__ == "__main__":
    main()
