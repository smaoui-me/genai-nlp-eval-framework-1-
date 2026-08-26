# Annotation product quickstart

This repository provides a reusable human-in-the-loop product, not only the
SciREX experiment. It accepts text, PDF, or a selected CSV ticket; optionally
suggests a destination department; extracts important information; records
human corrections; and exports traceable reviewed JSON.

## Prerequisites

- Docker Engine with Docker Compose v2 (Docker Desktop is sufficient)
- Access to an organization-approved OpenAI-compatible LLM endpoint

## Start with Docker Compose

From the repository root:

```bash
cp .env.example .env
# Edit .env and add the LLM endpoint, API key, and deployment name.
docker compose up --build
```

Open <http://localhost:8501>. Stop the application with `Ctrl+C`, or run it in
the background with `docker compose up --build -d` and stop it using
`docker compose down`.

The home page and configuration UI work without credentials. LLM buttons stay
disabled until an endpoint is configured. Therefore, credentials may be left
blank for an interface-only review.

Reviewed exports are persisted on the host under
`genai-nlp-annotation-tool/data/gold_exports/`.

## Run without Compose

Build and run the same image directly:

```bash
docker build --target runtime -t genai-nlp-annotation-tool:local .
docker run --rm -p 8501:8501 --env-file .env genai-nlp-annotation-tool:local
```

For Ollama running on the Docker host, set
`OLLAMA_ENDPOINT=http://host.docker.internal:11434/v1` in `.env`. On Linux,
also add `--add-host=host.docker.internal:host-gateway` to `docker run`.

## Optional development environments

The same project can also run directly with Python 3.11:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd genai-nlp-annotation-tool
python -m streamlit run app.py
```

VS Code users may instead open the repository through **Dev Containers: Reopen
in Container**. This is optional; Docker Compose is the portable default.

## Validate the installation

```bash
docker compose --profile test run --rm test
```

## Run evaluation scripts in Docker

The `evaluation` image contains the framework modules, scripts, prompts, and
configs. Dataset and result directories are mounted from the host so they are
not baked into a large image:

```bash
docker compose --profile tools build evaluation
docker compose --profile tools run --rm evaluation \
  python scripts/annotation/check_scirex_study_readiness.py
```

Replace the Python command with any entry point under `scripts/`. LLM-backed
runs read the same variables from `.env`; network-free preprocessing and
scoring commands do not require credentials.

## Suggested ten-minute walkthrough

1. Open **Annotate** and select **Internal support tickets**.
2. Keep the sample ticket or choose **Upload file**.
3. To test CSV mapping, upload `sample_data/tickets.csv`, select `text`
   as the main text column, `title` as context, and `ticket_id` as identifier.
4. Review the ticket entity labels and their generated prompt.
5. Keep optional department routing enabled and choose **LLM classifier** for
   the first run. Its generated prompt remains visible and editable.
6. Run pre-annotation.
7. Confirm or change the department, then confirm/relabel/delete extracted
   spans and add one missed span if appropriate.
8. Download the reviewed JSON. It contains the immutable model predictions,
   approved extraction labels, routing decision, prompts' hashes, source
   identifiers, token usage, and review history.

To try routing without an extra LLM call, select **Local embedding classifier**.
Upload `sample_data/classification/embedding_demo/reference_tickets.csv`, map
`ticket_id` to ID, `text` to text, and `gold_queue` to department, then click
**Build or replace index**. This uses a pretrained encoder and reviewed examples;
it builds an index but does not train or fine-tune a model.

## Adapt it to another use case

Copy one of the YAML files in
`genai-nlp-annotation-tool/config/use_cases/`, then edit:

- `name` and `description`
- `entity_labels`
- `label_definitions`
- `domain_guidance`
- optional routing departments and guidance

Restart or refresh the app. The new YAML file appears in the use-case selector;
no Python change is required.

## Current scope

Implemented now:

- Configurable extraction schemas and visible editable prompts
- Text, PDF, and interactive CSV-row input
- Optional LLM or local embedding department suggestion
- Human approval/correction of routing and extracted entities
- Traceable JSON export and annotation-quality diagnostics
- SciREX benchmark/evaluation workflow

Not yet a production ticketing integration:

- CSV tickets are reviewed one at a time; there is no connector or job queue
- No user authentication, roles, central database, or multi-tenant isolation
- Reviewer corrections do not automatically retrain the LLM
- Department accuracy requires an independently labeled internal test set

The recommended learning path is: collect consistent reviewer-approved tickets,
split them into reference, development, and held-out sets, build the index only
from the reference set, tune the neighbor count on development data, and compare
LLM and embedding routing once on the untouched held-out set. Only then consider
fine-tuning if enough high-quality data exists.

The optional retrieval implementation is available as the `embedding_rag`
classification method. Its separate Docker image, private-data layout, index
builder, leakage controls, and evaluation commands are documented in
[`docs/classification/EMBEDDING_RETRIEVAL.md`](docs/classification/EMBEDDING_RETRIEVAL.md).

## Security

Use the included synthetic tickets for the first review. Do not upload real
customer data to a public model, Codespace, or hosted demo without formal
approval. Gold exports remain in the mounted local workspace unless the user
downloads or moves them elsewhere.

Never commit `.env` or `.streamlit/secrets.toml`. For internal data, confirm
the endpoint's data-retention, geographic-processing, and access policies first.
