# Embedding retrieval for user-owned ticket data

The `embedding_rag` method retrieves reviewed historical tickets that are
semantically similar to a new ticket and inserts them as dynamic few-shot
examples for the LLM classifier.

The Streamlit app also exposes a simpler **Local embedding classifier** for
department routing. It retrieves reviewed tickets and predicts the department
with a similarity-weighted nearest-neighbor vote, so that routing path makes no
LLM call. Entity extraction remains LLM-based. The app records the encoder,
reference-data hash, retrieved ticket IDs, similarities, vote share, prediction,
and reviewer correction in the exported JSON.

This method does **not** train or fine-tune the LLM. It uses a pretrained local
embedding encoder to index labeled examples. It can later be extended with a
fine-tuned encoder, but that is a separate experiment requiring sufficient
reviewed data and a held-out evaluation set.

In the app, index construction is handled explicitly: select the local embedding
classifier, upload a reviewed CSV, map its ID/text/department columns, and click
**Build or replace index**. The mounted `data/classification/retrieval/` directory
persists the canonical reference CSV and index across container restarts.

## Required data

Prepare a reviewed reference CSV with this default schema:

```csv
ticket_id,text,gold_type,gold_queue,gold_tags
T-001,"VPN access fails after reset",Incident,IT Support,"[""VPN"",""Access""]"
```

- `ticket_id` must be unique.
- `text` contains the complete input used for retrieval.
- `gold_type` and `gold_queue` are single reviewed labels.
- `gold_tags` is a JSON list of zero or more reviewed tags.

Column names can be changed through the index builder's `--*-column` options.
For a department-only use case, place the department in `gold_queue`, use one
constant value such as `Request` for `gold_type`, and use `[]` for tags.

## Prevent evaluation leakage

Split the data before building the index:

```text
reviewed tickets
├── reference set  -> embedding index and retrieved examples
└── held-out set   -> final metrics only
```

Split by the unit that could otherwise leak information, such as customer,
incident, site, or time period—not merely by random rows. Never build the index
from the held-out file. Query-time ID and duplicate-text exclusion provide an
additional safeguard but cannot repair an invalid split.

## Build the optional Docker image

```bash
docker compose --profile embeddings build embeddings
```

The image is separate because Sentence Transformers installs PyTorch and is
substantially larger than the annotation application. The declared environment
uses the official CPU-only PyTorch wheel; no CUDA runtime is required. The
encoder model is downloaded on first use and retained in the
`embedding-model-cache` Docker volume.

## Build an index

For a safe functional test, use the synthetic reference file at
`sample_data/classification/embedding_demo/reference_tickets.csv` and follow
[`docs/TESTING.md`](../TESTING.md). The private path below is the recommended
layout for actual user-owned data.

Place the private reference file at
`data/classification/retrieval/reference_tickets.csv`, then run:

```bash
docker compose --profile embeddings run --rm embeddings \
  python scripts/classification/build_embedding_index.py \
  --input data/classification/retrieval/reference_tickets.csv \
  --output-dir data/classification/retrieval/index
```

The command writes normalized embeddings, canonical reference records, a
derived label schema, and a manifest containing source and artifact hashes.
These files are ignored by Git by default.

## Evaluate on held-out labeled tickets

```bash
docker compose --profile embeddings run --rm embeddings \
  python scripts/classification/run_ticket_classification_with_evidence.py \
  --method embedding_rag \
  --input data/classification/retrieval/held_out_tickets.csv \
  --labels data/classification/retrieval/index/allowed_labels.json \
  --run-name internal_embedding_rag
```

The normal classification metrics and error tables are written under
`results/classification/`. Each prediction also records retrieved ticket IDs,
similarity scores, encoder identity, and the reference-source hash.

Compare this result with `zero_shot` and a static `few_shot` run on the exact
same held-out rows. Do not select the final method using the held-out set;
perform prompt and top-k selection on a separate development split.

## Classify unlabeled tickets

For a CSV containing only an ID and text, add `--predict-only`:

```bash
docker compose --profile embeddings run --rm embeddings \
  python scripts/classification/run_ticket_classification_with_evidence.py \
  --method embedding_rag \
  --predict-only \
  --input data/classification/retrieval/unlabeled_tickets.csv \
  --labels data/classification/retrieval/index/allowed_labels.json \
  --output results/classification/internal_predictions.jsonl
```

If the input column names differ, copy
`configs/classification/embedding_rag.yaml`, update its `dataset.columns`, and
pass the copy through `--config`.

## Security and lifecycle

- Keep reference records, indexes, model caches, and outputs in approved storage.
- Confirm that sending retrieved examples to the configured LLM endpoint is
  permitted; retrieved examples can contain the same sensitive data as tickets.
- Rebuild the index when reviewed examples, labels, or the encoder model change.
- Retain an immutable held-out set for every reported comparison.
- Consider encoder or LLM fine-tuning only after label policy and reviewer
  agreement are stable and the retrieval baseline has been measured.
