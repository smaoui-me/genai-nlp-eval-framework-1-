# Data sources and reproducibility

This repository keeps compact fixtures, frozen evaluation manifests, cleaned
tables required by the application, and provenance metadata in ordinary Git.
Large raw downloads and regenerated intermediate JSONL files are excluded to
keep the public clone manageable and below GitHub's file-size limits.

No confidential or internal company data is included.

## Third-party datasets

| Dataset | Purpose | Upstream source | Upstream license |
|---|---|---|---|
| SciREX | Long-document scientific entity extraction | <https://github.com/allenai/SciREX> | Apache-2.0 |
| Few-NERD | NER extraction baselines | <https://huggingface.co/datasets/DFKI-SLT/few-nerd> | CC BY-SA 4.0 |
| Customer Support Tickets | Synthetic ticket classification | <https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets> | CC BY-NC 4.0 |

The MIT license at the repository root applies to project code, not to these
datasets. Users remain responsible for complying with each upstream license,
including attribution, share-alike, and noncommercial restrictions.

## Public data retained here

- `annotation/fixtures/scirex/`: eight-example network-free smoke fixture.
- `annotation/processed/scirex/csv/`: linked 1,000-window SciREX export used by
  the documented study. Join on `example_id`; group and split on `doc_id`.
- `annotation/processed/scirex/{manifest,statistics,label_schema}.json`:
  preprocessing provenance and validation summaries.
- `extraction/processed/fewnerd_location_test_1000.csv`: deterministic
  location-only evaluation slice. Smaller experiments use `--limit`.
- `classification/processed/ticket_extraction_eval.csv`: fixed first 1,000
  cleaned synthetic tickets despite its retained legacy filename.
- `../eval/corpora/`: frozen development, held-out, and operational SciREX
  manifests.

## Reconstruct SciREX intermediates

Download `release_data.tar.gz` from the official SciREX repository and extract
`train.jsonl`, `dev.jsonl`, and `test.jsonl` into
`data/annotation/raw/release_data/`. Then run:

```bash
python scripts/annotation/prepare_scirex.py \
  --raw-dir data/annotation/raw/release_data \
  --output-dir data/annotation/processed/scirex \
  --config configs/annotation/scirex_preprocessing.yaml \
  --overwrite
python scripts/annotation/validate_scirex.py \
  --processed-dir data/annotation/processed/scirex
python scripts/annotation/export_scirex_csv.py
```

The committed manifest records source and output SHA-256 hashes. A regenerated
artifact should match those hashes when the same upstream release and config
are used.

## Reconstruct Few-NERD data

Download the `intra` Parquet split from the upstream Hugging Face dataset into
`data/extraction/raw/intra/`. Convert the test file and build the public slice:

```bash
python scripts/extraction/convert_parquet_to_csv.py \
  data/extraction/raw/intra/test-00000-of-00001.parquet
python scripts/extraction/build_fewnerd_eval.py \
  --input data/extraction/raw/intra/test-00000-of-00001.csv \
  --output data/extraction/processed/fewnerd_location_test_1000.csv \
  --limit 1000
```

## Reconstruct ticket-classification data

Download `aa_dataset-tickets-multi-lang-5-2-50-version.csv` from the upstream
dataset, save it as `data/classification/raw/customer_support_tickets.csv`, and
run:

```bash
python scripts/classification/build_ticket_classification_sample.py --limit 1000
```

This source is synthetic and licensed for noncommercial use. It is not an
estimate of performance on an organization's internal ticket distribution.

## User-owned classification references

Private reviewed tickets and embedding indexes belong under
`classification/retrieval/` and are ignored by Git. The repository contains no
company ticket data or pretrained index. Follow
[`docs/classification/EMBEDDING_RETRIEVAL.md`](../docs/classification/EMBEDDING_RETRIEVAL.md)
to create a leakage-safe reference index and evaluate it on a separate held-out
set.
