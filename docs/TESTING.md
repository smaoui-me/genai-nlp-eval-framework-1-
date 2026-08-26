# Complete project test guide

This guide uses Windows PowerShell and the docker-compose command available on
this development machine. With newer Docker versions, docker compose can be
used instead. Run all commands from the repository root.

## 1. Configure the LLM

    Copy-Item .env.example .env
    notepad .env

Keep the working values from the approved provider:

    LLM_ENDPOINT=<OpenAI-compatible base URL>
    LLM_API_KEY=<secret>
    LLM_DEPLOYMENT_NAME=<deployment or model>

For local Ollama on the host, use:

    LLM_ENDPOINT=http://host.docker.internal:11434/v1
    LLM_API_KEY=ollama
    LLM_DEPLOYMENT_NAME=qwen3:4b

Never commit .env.

## 1a. Exercise both routing methods in the app

Start the default embedding-capable application:

    docker-compose up --build annotation-tool

Open http://localhost:8501 and choose **Annotate**. Entity extraction always
uses the selected LLM. The optional department classifier has two independent
choices:

1. Select **LLM classifier**, inspect the editable routing prompt, and run one
   sample ticket. This adds one routing LLM call.
2. Select **Local embedding classifier**, upload
   `sample_data/classification/embedding_demo/reference_tickets.csv`, map
   `ticket_id`, `text`, and `gold_queue`, and build the index.
3. Run a ticket and inspect **Retrieved reviewed tickets** during review. This
   routing method uses zero LLM calls; only extraction consumes LLM calls.
4. Correct or approve the department and save the gold JSON. Verify that its
   classification block contains the classifier name, retrieved IDs and
   similarities, original prediction, and reviewer-approved department.

The index is persisted under `data/classification/retrieval/`. Building it is
not training: the pretrained encoder is unchanged. For a defensible internal
comparison, build from a reference split, choose `top_k` on a separate
development split, and report both methods once on an untouched held-out split.

## 2. Build and run network-free tests

    docker-compose build annotation-tool
    docker-compose --profile tools build evaluation
    docker-compose --profile test build test
    docker-compose --profile embeddings build embeddings
    docker-compose --profile test run --rm test

All ordinary tests must pass. The large-data SciREX test can be skipped because
the complete generated corpus is intentionally not baked into the image.

Now verify the LLM configuration without printing the key:

    docker-compose --profile tools run --rm evaluation python -c "import os; v=[os.getenv('LLM_ENDPOINT'),os.getenv('LLM_API_KEY'),os.getenv('LLM_DEPLOYMENT_NAME')]; assert all(v) and 'your-' not in ' '.join(v); print('Configured:',v[0],v[2])"

## 3. Test embedding retrieval without an LLM

Four disjoint synthetic files are provided in
sample_data/classification/embedding_demo/. Build only from the reference
file:

    docker-compose --profile embeddings run --rm embeddings python scripts/classification/build_embedding_index.py --input sample_data/classification/embedding_demo/reference_tickets.csv --output-dir data/classification/retrieval/index

The first run downloads the encoder but does not call an LLM. Inspect the four
generated artifacts:

    Get-ChildItem data/classification/retrieval/index
    Get-Content data/classification/retrieval/index/manifest.json
    Get-Content data/classification/retrieval/index/allowed_labels.json

Expected files are embeddings.npy, records.jsonl, manifest.json, and
allowed_labels.json. Query the index:

    docker-compose --profile embeddings run --rm embeddings python scripts/classification/query_embedding_index.py --index-dir data/classification/retrieval/index --text "The VPN is down and remote workers cannot connect" --top-k 3

The JSON must contain three hits with ticket IDs, similarities, and reviewed
labels. A leading hit should concern VPN, network access, or an outage. No fixed
similarity threshold is required.

## 4. Smoke-test all classification methods

These commands process one development ticket. Zero-shot, few-shot, and
embedding RAG make one LLM call; the agent makes two.

    docker-compose --profile tools run --rm evaluation python scripts/classification/run_ticket_classification_with_evidence.py --method zero_shot --input sample_data/classification/embedding_demo/development_tickets.csv --labels data/classification/retrieval/index/allowed_labels.json --limit 1 --run-name smoke_zero_shot

    docker-compose --profile tools run --rm evaluation python scripts/classification/run_ticket_classification_with_evidence.py --method few_shot --input sample_data/classification/embedding_demo/development_tickets.csv --labels data/classification/retrieval/index/allowed_labels.json --limit 1 --run-name smoke_few_shot

    docker-compose --profile tools run --rm evaluation python scripts/classification/run_ticket_classification_with_evidence.py --method agent_two_step --input sample_data/classification/embedding_demo/development_tickets.csv --labels data/classification/retrieval/index/allowed_labels.json --limit 1 --run-name smoke_agent

    docker-compose --profile embeddings run --rm embeddings python scripts/classification/run_ticket_classification_with_evidence.py --method embedding_rag --input sample_data/classification/embedding_demo/development_tickets.csv --labels data/classification/retrieval/index/allowed_labels.json --limit 1 --run-name smoke_embedding_rag

Inspect the embedding result:

    Get-Content results/classification/smoke_embedding_rag.jsonl -TotalCount 1
    Import-Csv results/classification/evaluation/smoke_embedding_rag_scores.csv | Format-List

It must have no error, contain a prediction, and list reference IDs and
similarities under retrieval.hits.

## 5. Compare classification methods on held-out tickets

Use development_tickets.csv for prompt and top-k choices. After choices are
fixed, run the untouched held-out file:

    docker-compose --profile tools run --rm evaluation python scripts/classification/run_ticket_classification_with_evidence.py --method zero_shot --input sample_data/classification/embedding_demo/held_out_tickets.csv --labels data/classification/retrieval/index/allowed_labels.json --limit 0 --run-name demo_zero_shot

    docker-compose --profile tools run --rm evaluation python scripts/classification/run_ticket_classification_with_evidence.py --method few_shot --input sample_data/classification/embedding_demo/held_out_tickets.csv --labels data/classification/retrieval/index/allowed_labels.json --limit 0 --run-name demo_few_shot

    docker-compose --profile tools run --rm evaluation python scripts/classification/run_ticket_classification_with_evidence.py --method agent_two_step --input sample_data/classification/embedding_demo/held_out_tickets.csv --labels data/classification/retrieval/index/allowed_labels.json --limit 0 --run-name demo_agent

    docker-compose --profile embeddings run --rm embeddings python scripts/classification/run_ticket_classification_with_evidence.py --method embedding_rag --input sample_data/classification/embedding_demo/held_out_tickets.csv --labels data/classification/retrieval/index/allowed_labels.json --limit 0 --run-name demo_embedding_rag

    docker-compose --profile tools run --rm evaluation python scripts/classification/compare_ticket_classification_with_evidence.py --pattern "demo_*_scores.csv" --output results/classification/evaluation/demo_method_comparison.csv
    Import-Csv results/classification/evaluation/demo_method_comparison.csv | Format-Table

This small comparison uses 12 examples and 60 logical calls: 12 each for
zero-shot, few-shot, and embedding RAG, plus 24 for the two-step agent. It proves
functionality but is not a statistically strong performance study.

Review type and queue accuracy/macro F1, tag micro precision/recall/F1, JSON
validity, label validity, and evidence validity.

## 6. Test prediction-only classification

    docker-compose --profile embeddings run --rm embeddings python scripts/classification/run_ticket_classification_with_evidence.py --method embedding_rag --predict-only --input sample_data/classification/embedding_demo/unlabeled_tickets.csv --labels data/classification/retrieval/index/allowed_labels.json --limit 0 --output results/classification/demo_unlabeled_predictions.jsonl
    Get-Content results/classification/demo_unlabeled_predictions.jsonl

Metrics are deliberately skipped because these rows have no independent gold.

## 7. Smoke-test all Few-NERD extraction methods

Five examples keep the test inexpensive. The public few-shot fixture removes
the previous dependency on a large ignored training download.

    docker-compose --profile tools run --rm evaluation python scripts/extraction/run_zero_shot_structured.py --limit 5 --run-name smoke_zero_shot_structured
    docker-compose --profile tools run --rm evaluation python scripts/extraction/run_zero_shot_freeform.py --limit 5 --run-name smoke_zero_shot_freeform
    docker-compose --profile tools run --rm evaluation python scripts/extraction/run_few_shot_structured.py --limit 5 --run-name smoke_few_shot_structured
    docker-compose --profile tools run --rm evaluation python scripts/extraction/run_few_shot_freeform.py --limit 5 --run-name smoke_few_shot_freeform
    docker-compose --profile tools run --rm evaluation python scripts/extraction/run_agent_verify_extraction.py --limit 5 --run-name smoke_agent_verify

    docker-compose --profile tools run --rm evaluation python scripts/extraction/compare.py --pattern "smoke_*_scores.csv" --output results/extraction/evaluation/smoke_method_comparison.csv
    Import-Csv results/extraction/evaluation/smoke_method_comparison.csv | Format-Table

Review strict and lenient F1, JSON validity, invalid-label rate, and the
per-sentence files under results/extraction/evaluation/.

## 8. Test SciREX long-document mechanics

On a fresh public clone, reconstruct the ignored runnable JSONL from the three
public linked CSVs. Skip this command when the file already exists:

    docker-compose --profile tools run --rm evaluation python scripts/annotation/rebuild_scirex_jsonl_from_csv.py

The reconstructed artifact has every field required by the runner and
evaluator. The public CSV export omits section spans, so sections arrays are
empty and the file hash differs from the raw-data preprocessing artifact.

Then run the network-free readiness and dry-run checks:

    docker-compose --profile tools run --rm evaluation python scripts/annotation/check_scirex_study_readiness.py
    docker-compose --profile tools run --rm evaluation python scripts/annotation/run_scirex_batch.py --config configs/annotation/scirex_operational_100.yaml --dry-run
    docker-compose --profile tools run --rm evaluation python scripts/annotation/run_scirex_batch.py --config configs/annotation/scirex_full_1000.yaml --dry-run

Make exactly one live sentence-level call:

    docker-compose --profile tools run --rm evaluation python scripts/annotation/run_scirex_batch.py --input eval/corpora/scirex_dev_pilot_20.jsonl --output results/annotation/scirex_smoke/predictions.jsonl --checkpoint-dir results/annotation/scirex_smoke/checkpoints --method few_shot_structured --prompt-mode suggested --limit 1 --max-sentences 1 --workers 1 --max-calls 1
    Get-Content results/annotation/scirex_smoke/predictions.jsonl

This is a connectivity/checkpoint test, not a long-document accuracy result.
Use docs/annotation/SCIREX_1000_RUNBOOK.md before paid study runs.

## 9. Test the annotation app

    docker-compose up --build -d
    docker-compose ps
    docker-compose logs --tail 100 annotation-tool

Open http://localhost:8501, then:

1. Select the internal support-ticket use case.
2. Paste a ticket from the synthetic unlabeled file.
3. Confirm the generated prompt is visible and editable.
4. Select the configured hosted or Ollama model.
5. Run pre-annotation and optional ticket routing.
6. Correct or add an entity, approve it, and export reviewed JSON.
7. Confirm a file appears in genai-nlp-annotation-tool/data/gold_exports/.
8. Confirm the evaluation and model-comparison pages render.

Stop it with:

    docker-compose down

## 10. Use company data

Create four files with the same columns as the synthetic fixtures:

| Split | Purpose |
|---|---|
| Reference | Reviewed historical tickets used by retrieval |
| Development | Prompt, encoder, and top-k selection |
| Held-out test | One-time final method comparison |
| Unlabeled | New tickets for prediction |

Keep related customers, incidents, sites, and time periods in one split. Never
index development, held-out, or unlabeled files. Rebuild after changing
reference texts, labels, or encoder.

The 20-ticket reference fixture proves functionality only. A company study
needs enough independently reviewed examples per department and issue type to
measure imbalance and rare failures, with uncertainty intervals and per-class
error analysis.

## Pass criteria

- Docker tests pass.
- All four embedding artifacts exist.
- A no-LLM query returns sensible hits.
- All classification and extraction methods finish without row errors.
- Embedding predictions include retrieval provenance.
- Both SciREX dry runs and the one-call smoke test succeed.
- The app pre-annotates, accepts a correction, and persists an export.
