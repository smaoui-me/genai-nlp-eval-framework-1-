# NLP Annotation Tool

> **TLDR:** A local Streamlit tool that turns raw text/PDF into gold-standard NER data — an LLM pre-labels entities, a human confirms/fixes/deletes them, and the result exports as gold JSON.

## Flow

```
Raw text → LLM pre-annotation → Human correction → Gold-standard JSON
```

Pre-annotation reuses the four extraction strategies benchmarked in the
[genai-nlp-eval-framework-1-](https://github.com/smaoui-me/genai-nlp-eval-framework-1-)
project (zero-shot / few-shot × freeform / structured JSON output),
generalized here to an arbitrary, user-defined label set instead of the
framework's single "location" label.

## Pages

- **Home** — flow overview + a static mockup of the review UI
- **Annotate** — upload a `.txt`/`.pdf` (or pick a sample), choose labels + method,
  run pre-annotation, review the highlighted spans, confirm/relabel/delete or add
  missed entities, export gold JSON (download or save to `data/gold_exports/`)
- **Extraction Method Evaluation** — precision/recall/F1, JSON validity, and
  invalid-label rate per method, loaded from the eval-framework's benchmark CSVs
- **Annotation Evaluation** — acceptance / edit / deletion rates computed from the
  saved review logs — a live measure of how much manual work each method saves

## Features

- Zero-shot & few-shot pre-labeling, freeform or structured JSON output
- Works on uploaded `.txt` / `.pdf` or built-in sample tickets
- User-defined entity labels (not hardcoded to one type)
- Character-offset-accurate span highlighting and gold export
- Full audit trail: every span's source (model/human) and status (confirmed/
  edited/deleted) is preserved for evaluation

## Live demo

The app is deployed on Streamlit Community Cloud, so it can be tried without
installing anything:

**➡️ https://genai-nlp-annotation-tool-v1.streamlit.app**

The deployed app uses a shared LLM key configured centrally in the Cloud
app's Secrets (not stored in this repo), so it works out of the box.

## Run locally (for development)

```bash
git clone <this-repo-url>
cd genai-nlp-annotation-tool

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Add LLM credentials
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with LLM_ENDPOINT / LLM_API_KEY / LLM_DEPLOYMENT_NAME

streamlit run app.py
```

## Deploying / updating the Streamlit Community Cloud app

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub.
3. **New app** → pick this repo, branch `main`, main file `app.py`.
4. Under **Settings → Secrets**, paste the contents of the local
   `.streamlit/secrets.toml` (same three keys).
5. Deploy. Every push to `main` auto-redeploys, so this setup only needs to
   happen once.

**Seeing "No LLM credentials found"?** The Secrets dashboard is empty or the app
hasn't rebooted since you saved it. Go to the app on
[share.streamlit.io](https://share.streamlit.io) → **⋮ menu → Settings → Secrets**,
paste the three `LLM_*` keys, save, then **Reboot app** from the same menu.

## Data & credentials

- `data/gold_exports/` stores reviewed annotations as JSON (gitignored) — this is
  what powers the Annotation Evaluation page. On Streamlit Cloud this resets on
  redeploy/reboot (no persistent disk), so download gold JSON exports you want
  to keep.
- The Extraction Method Evaluation page reads benchmark CSVs produced by the
  [genai-nlp-eval-framework-1-](https://github.com/smaoui-me/genai-nlp-eval-framework-1-)
  project. That project isn't part of this repo, so those CSVs are bundled
  into `data/eval_benchmarks/` here instead — see `utils/evaluation_data.py`.
- Credentials come from `st.secrets` (`.streamlit/secrets.toml` locally, the
  Cloud dashboard when deployed). `.streamlit/secrets.toml` and `.env` are
  never committed — see `.streamlit/secrets.toml.example` / `.env.example`.

---

Built with [Streamlit](https://streamlit.io)
