# GenAI NLP Evaluation and Annotation Framework

A reproducible framework for evaluating LLM-based classification and entity
extraction, generating pre-annotations, reviewing model suggestions, and
exporting traceable gold data.

The repository contains two connected components:

- A Streamlit annotation product for configurable extraction, optional ticket
  routing, human review, and reviewed JSON export.
- Offline evaluation pipelines for ticket classification, Few-NERD extraction,
  and long-document SciREX experiments.

The product is research software. It does not yet provide authentication,
role-based access, a central database, work queues, or multi-tenant isolation.

## Start the annotation product

Requirements: Docker Engine and Docker Compose v2.

```bash
cp .env.example .env
docker compose up --build
```

Open <http://localhost:8501>. The interface can be inspected without an LLM.
Pre-annotation and routing are enabled after a hosted OpenAI-compatible endpoint
or local Ollama model is configured in `.env`.

Reviewed exports are stored outside the image at
`genai-nlp-annotation-tool/data/gold_exports/`.

See [QUICKSTART.md](QUICKSTART.md) for the complete walkthrough.
For an end-to-end verification of every method, including synthetic embedding
data and expected outputs, follow [docs/TESTING.md](docs/TESTING.md).

## Use the product with another domain

1. Copy a YAML template under
   `genai-nlp-annotation-tool/config/use_cases/`.
2. Define the entity labels, label descriptions, domain instructions, and
   optional routing departments.
3. Restart the application and select the new use case.
4. Inspect and edit the generated prompt before inference.
5. Review every suggested span before treating the export as gold data.

No Python changes are required for a new extraction schema.

## Validate the repository in Docker

```bash
docker compose --profile test run --rm test
docker compose up --build -d
docker compose ps
docker compose down
```

The first command builds the dedicated test target and runs the network-free
test suite. By default, the application uses `runtime-embeddings`, which supports
both LLM and local embedding routing. Set `ANNOTATION_TARGET=runtime` in `.env`
only when a smaller LLM-only image is preferred.

## Repository structure

| Path | Purpose |
|---|---|
| `genai-nlp-annotation-tool/` | Streamlit product, domain templates, prompts, and product tests |
| `src/genai_eval/` | Reusable evaluation and preprocessing modules |
| `scripts/` | Reproducible classification, extraction, and SciREX entry points |
| `configs/` and `prompts/` | Versioned experiment definitions |
| `data/` | Compact public fixtures, cleaned evaluation tables, and provenance metadata |
| `eval/corpora/` | Frozen SciREX study manifests |
| `docs/annotation/` | Study protocols and completed evaluation summaries |
| `docs/report/` | LaTeX source and compiled scientific report |
| `tests/` | Framework and container-distribution tests |

Large raw datasets and generated prediction files are intentionally excluded
from ordinary Git. See [data/README.md](data/README.md) for sources, licenses,
and reconstruction commands. Runtime output conventions are documented in
[results/README.md](results/README.md).

## Offline evaluation

The full evaluation environment is also available through Compose. Local
`data/`, `eval/`, and `results/` directories are mounted into the container, so
large datasets and generated outputs remain outside the image:

```bash
docker compose --profile tools build evaluation
docker compose --profile tools run --rm evaluation \
  python scripts/annotation/check_scirex_study_readiness.py
```

To work outside Docker, install the runtime and development dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Representative entry points:

```bash
python scripts/classification/run_ticket_classification_with_evidence.py --method zero_shot --limit 1
python scripts/extraction/run_few_shot_structured.py --limit 10
python scripts/annotation/check_scirex_study_readiness.py
```

The app can also route tickets directly with a local similarity-weighted vote
over reviewed examples. The offline `embedding_rag` experiment uses those
examples as dynamic few-shot context for an LLM. Both require the larger
PyTorch-based dependency stack. See
[docs/classification/EMBEDDING_RETRIEVAL.md](docs/classification/EMBEDDING_RETRIEVAL.md)
for the data contract, leakage-safe split, index build, evaluation, and
prediction-only workflow.

LLM runs require the variables shown in `.env.example`. Tests do not make
network calls.

## Research artifacts

The completed report is available as
[NLP_Lab_Project_Report.pdf](NLP_Lab_Project_Report.pdf). Its LaTeX source is
kept under [`docs/report/`](docs/report/). The retained study documentation
distinguishes prompt-development results, held-out evaluation, and the
100-paper operational run. The configured 1,000-window SciREX run is prepared
but is not presented as a completed experiment.

## Security and privacy

Do not send confidential tickets or documents to an endpoint that has not been
approved for the relevant data. Secrets are loaded at runtime and are excluded
from Git and the Docker build context. See [SECURITY.md](SECURITY.md).

## Contributing and license

Development and validation guidance is in
[CONTRIBUTING.md](CONTRIBUTING.md). Repository code is released under the MIT
License. Third-party datasets retain their own licenses; see
[data/README.md](data/README.md).
