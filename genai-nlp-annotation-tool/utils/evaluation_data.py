"""
Loads the pre-computed benchmark scores (precision, recall, F1, ...) for
each LLM pre-labeling method, so the Extraction Method Evaluation page can
draw charts without calling the LLM itself.

The scores come from https://github.com/smaoui-me/genai-nlp-eval-framework-1-,
which ran each method against a labeled test set (FewNERD) and saved the
results as CSV files. We keep a copy of those CSVs in this repo (see
data/eval_benchmarks/), so the page still works when deployed on Streamlit
Community Cloud, where that other project isn't available. If that project
*is* checked out locally with fresher results, we use those instead.

pandas (imported as `pd`) is a library for working with tables of data. A
"DataFrame" is pandas' name for one such table.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

_APP_DIR = Path(__file__).resolve().parent.parent
_BUNDLED_EVAL_DIR = _APP_DIR / "data" / "eval_benchmarks" / "extraction_evaluation"
_LOCAL_EVAL_FRAMEWORK_DIR = _APP_DIR.parent / "genai-nlp-eval-framework-1-" / "results" / "extraction" / "evaluation"

# Prefer a locally checked-out eval-framework project's results (may be
# newer), otherwise fall back to our bundled copy.
EVAL_DIR = _LOCAL_EVAL_FRAMEWORK_DIR if _LOCAL_EVAL_FRAMEWORK_DIR.exists() else _BUNDLED_EVAL_DIR

# Friendlier names to show in charts instead of the raw method IDs.
METHOD_DISPLAY_NAMES = {
    "few_shot_freeform": "Few-shot — freeform",
    "few_shot_structured": "Few-shot — structured JSON",
    "few_shot": "Few-shot — multi-label (legacy)",
    "zero_shot_freeform": "Zero-shot — freeform",
    "zero_shot_structured": "Zero-shot — structured JSON",
    "zero_shot": "Zero-shot — multi-label (legacy)",
}

# These two ran before the eval-framework was refactored into the four
# methods above (they predict many fine-grained types, not just one). They
# aren't an option on the Annotate page, so we keep them out of the main
# method-comparison charts and only offer them in the per-entity-type
# breakdown, where they're still the only methods with anything to show.
LEGACY_METHODS = {"few_shot", "zero_shot"}


def is_available() -> bool:
    """True if we can find a folder of benchmark results."""
    return EVAL_DIR.exists()


def load_scores(include_legacy: bool = True) -> pd.DataFrame:
    """Combine every method's "*_scores.csv" into one table, one row per method.
    Pass include_legacy=False to drop the two pre-refactor legacy runs."""
    if not EVAL_DIR.exists():
        return pd.DataFrame()

    rows = []
    for csv_path in sorted(EVAL_DIR.glob("*_scores.csv")):  # finds e.g. "few_shot_scores.csv"
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        rows.append(df.iloc[0])  # each file only has one row of overall scores

    if not rows:
        return pd.DataFrame()

    scores = pd.DataFrame(rows).reset_index(drop=True)
    scores["display_name"] = scores["method"].map(lambda m: METHOD_DISPLAY_NAMES.get(m, m))

    if not include_legacy:
        scores = scores[~scores["method"].isin(LEGACY_METHODS)].reset_index(drop=True)

    return scores


def load_per_type(method: str) -> pd.DataFrame | None:
    """Load precision/recall/F1 broken down by entity type for one method
    (e.g. "location" vs "person" separately). None if not available."""
    path = EVAL_DIR / f"{method}_per_type.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df[df["support"] > 0].sort_values("support", ascending=False)


def _summarize_spans(raw: str) -> str:
    """Turn a saved-as-text list of span dicts (e.g. "[{'text': 'Berlin',
    'type': 'location-GPE', ...}]") into a short readable string like
    "Berlin (location-GPE), Munich (location-GPE)", for a much smaller and
    easier-to-read table cell than the raw Python data."""
    try:
        spans = ast.literal_eval(raw)  # safely parse the Python-literal text back into real list/dict objects
    except (ValueError, SyntaxError):
        return str(raw)
    if not isinstance(spans, list):
        return str(raw)
    return ", ".join(f"{s.get('text', '?')} ({s.get('type', '?')})" for s in spans if isinstance(s, dict)) or "—"


def load_errors(method: str, limit: int = 25) -> pd.DataFrame | None:
    """Load the worst-scoring individual examples for one method, with the
    gold/predicted spans reformatted into short readable text."""
    path = EVAL_DIR / f"{method}_errors.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path).sort_values("strict_f1").head(limit).copy()
    df["gold_spans"] = df["gold_spans"].map(_summarize_spans)
    df["pred_spans"] = df["pred_spans"].map(_summarize_spans)
    return df
