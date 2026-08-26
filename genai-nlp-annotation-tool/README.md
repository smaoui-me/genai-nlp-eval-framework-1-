# Annotation product

This directory contains the Streamlit human-in-the-loop annotation product.
Run it from the repository root with Docker Compose; the canonical setup and
dependency files are intentionally maintained at the root.

## Workflow

1. Select a YAML-defined use case.
2. Enter text or select a row from an uploaded TXT, PDF, or CSV file.
3. Inspect and optionally edit the schema-derived extraction prompt.
4. Optionally classify the destination with an editable LLM prompt or a local
   embedding index built from reviewed tickets.
5. Run LLM pre-annotation.
6. Confirm, relabel, delete, or add spans.
7. Export the reviewed annotations and complete audit history as JSON.

## Main modules

| Path | Purpose |
|---|---|
| `app.py` | Streamlit entry point and page navigation |
| `pages/1_annotate.py` | Input, inference, review, and export workflow |
| `pages/2_annotation_evaluation.py` | Review-log and routing diagnostics |
| `pages/3_model_comparison.py` | Side-by-side model disagreement analysis |
| `config/use_cases/` | Reusable label schemas and domain instructions |
| `utils/` | LLM access, parsing, validation, storage, and evaluation helpers |
| `tests/` | Network-free product tests |

## Configuration

Hosted endpoints use the `LLM_ENDPOINT`, `LLM_API_KEY`, and
`LLM_DEPLOYMENT_NAME` variables defined in the root `.env.example`. A local
Ollama server can be exposed through `OLLAMA_ENDPOINT` and `OLLAMA_MODELS`.

The default Docker image also includes Sentence Transformers. In the Annotate
page, select **Local embedding classifier**, upload a reviewed reference CSV,
map its ID/text/department columns, and build the index. This is nearest-neighbor
classification, not model training. The smaller `runtime` image can be selected
through `ANNOTATION_TARGET=runtime`, but it disables embedding routing.

The product treats model output as a suggestion. Uploaded text does not contain
gold labels unless the source dataset supplies them separately. Gold data is
created only after reviewer approval or imported from an independently
annotated benchmark.

## Run and test

From the repository root:

```bash
docker compose up --build
docker compose --profile test run --rm test
```

See the root [README](../README.md) and [quickstart](../QUICKSTART.md) for full
instructions and current product limitations.
