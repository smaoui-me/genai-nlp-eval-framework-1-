"""Query a built embedding index without making an LLM call."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from genai_eval.classification.retrieval.embedding_index import (
    EmbeddingIndex,
    encode_texts,
    load_encoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show the nearest reviewed tickets in an embedding index."
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/classification/retrieval/index"),
    )
    parser.add_argument("--text", required=True, help="Ticket text to query.")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    index = EmbeddingIndex(args.index_dir)
    encoder = load_encoder(index.encoder_model)
    vector = encode_texts(
        encoder,
        [args.text],
        batch_size=1,
        show_progress_bar=False,
    )[0]
    hits = index.query_by_vector(vector, top_k=args.top_k, exclude_text=args.text)
    output = {
        "query": args.text,
        "encoder_model": index.encoder_model,
        "index_source_sha256": index.manifest["source"]["sha256"],
        "hits": [
            {
                "rank": rank,
                "ticket_id": hit.ticket_id,
                "similarity": round(hit.similarity, 6),
                "type": hit.type_label,
                "queue": hit.queue_label,
                "tags": hit.tags,
                "text": hit.text,
            }
            for rank, hit in enumerate(hits, start=1)
        ],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
