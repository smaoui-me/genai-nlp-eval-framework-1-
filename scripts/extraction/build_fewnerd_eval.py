"""Build a processed FewNERD extraction evaluation dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from genai_eval.extraction.labels import ALLOWED_FINE_ENTITY_TYPES, COARSE_LABELS, FINE_LABELS
from genai_eval.extraction.utils import build_spans, parse_int_array, parse_token_array, to_json_string


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    processed_rows = []
    for _, row in df.iterrows():
        tokens = parse_token_array(row["tokens"])
        coarse_tags = parse_int_array(row["ner_tags"])
        fine_tags = parse_int_array(row["fine_ner_tags"])
        if len(tokens) != len(fine_tags):
            raise ValueError(f"Token/tag length mismatch for id={row['id']}")
        processed_rows.append(
            {
                "id": row["id"],
                "tokens": to_json_string(tokens),
                "spans": to_json_string(build_spans(tokens, fine_tags, FINE_LABELS)),
                "coarse_spans": to_json_string(build_spans(tokens, coarse_tags, COARSE_LABELS)),
            }
        )
    return pd.DataFrame(processed_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FewNERD extraction eval data.")
    parser.add_argument("--input", default="data/extraction/raw/intra/test-00000-of-00001.csv")
    parser.add_argument("--output", default="data/extraction/processed/fewnerd_intra_test.csv")
    parser.add_argument("--labels-output", default="data/extraction/processed/fewnerd_allowed_entity_types.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    labels_output_path = Path(args.labels_output)

    df = pd.read_csv(input_path)
    processed = process_dataframe(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_output_path.parent.mkdir(parents=True, exist_ok=True)

    processed.to_csv(output_path, index=False)
    labels_output_path.write_text(
        json.dumps({"entity_types": ALLOWED_FINE_ENTITY_TYPES}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved {len(processed)} rows to {output_path}")
    print(f"Saved allowed entity types to {labels_output_path}")


if __name__ == "__main__":
    main()
