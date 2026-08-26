"""
Model comparison — run two or more models on the same text and see where
they disagree.

Two reasons this page is useful:

1. **Deciding whether a cheaper model is good enough.** Without gold labels
   you cannot measure accuracy, but you can measure how far a cheap model
   drifts from a stronger one. If they agree on 95% of spans, the difference
   is unlikely to matter for pre-annotation, and pre-annotation is a job where
   a human checks the output anyway.
2. **Finding rows worth reviewing.** Disagreement between models is a
   confidence signal in its own right — the same idea as the "Model agreement"
   option on the Annotate page, shown here span by span.

The overlap number is Jaccard: shared spans divided by the union of both
models' spans. Plain agreement would flatter a model that finds almost
nothing, since the spans it never proposed would not count against it.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.extraction_methods import METHODS, Pass, run_annotation
from utils.file_utils import extract_text
from utils.labels import DEFAULT_LABELS
from utils.llm_client import is_configured, missing_credentials_message
from utils.model_registry import available_choices, choice_by_id
from utils.sample_data import SAMPLES
from utils.uncertainty import agreement_report, span_key

st.title(":material/compare_arrows: Model comparison")
st.caption("Run the same text through two or more models and see where they differ")

if not is_configured():
    st.warning(missing_credentials_message(), icon=":material/warning:")
    st.stop()

choices = available_choices()
if len(choices) < 2:
    st.info(
        "Only one model is configured, so there is nothing to compare yet. Add another endpoint in "
        "`.streamlit/secrets.toml` (see `utils/model_registry.py` for the format) — a local "
        "[Ollama](https://ollama.com) install works and is free.",
        icon=":material/info:",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.subheader(":material/tune: Setup", anchor=False)

    col_src, col_models = st.columns([2, 2])
    with col_src:
        source_mode = st.radio("Text", ["Sample document", "Upload file", "Paste text"], key="cmp_source")
        if source_mode == "Sample document":
            sample_name = st.selectbox("Sample", list(SAMPLES.keys()), key="cmp_sample")
            doc_text = SAMPLES[sample_name]
        elif source_mode == "Upload file":
            uploaded = st.file_uploader("Upload .txt or .pdf", type=["txt", "pdf"], key="cmp_upload")
            doc_text = extract_text(uploaded) if uploaded is not None else ""
        else:
            doc_text = st.text_area("Paste text", height=160, max_chars=20_000, key="cmp_paste")

    with col_models:
        ids = [c.id for c in choices]
        labels_by_id = {c.id: c.label for c in choices}
        selected_ids = st.multiselect(
            "Models to compare",
            options=ids,
            default=ids[:2],
            format_func=lambda i: labels_by_id.get(i, i),
            key="cmp_models",
        )
        method_id = st.selectbox(
            "Method",
            options=list(METHODS.keys()),
            format_func=lambda m: METHODS[m]["label"],
            key="cmp_method",
        )
        entity_labels = st.multiselect(
            "Entity labels", options=sorted(set(DEFAULT_LABELS)), default=list(DEFAULT_LABELS), key="cmp_labels"
        )
        max_sentences = st.slider("Max sentences", 1, 15, 4, key="cmp_max_sentences")

    run = st.button(
        ":material/play_arrow: Run comparison",
        type="primary",
        width="stretch",
        disabled=not doc_text or len(selected_ids) < 2 or not entity_labels,
        key="cmp_run",
    )

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run:
    status = st.status("Running each model over the text...", expanded=True)
    try:
        passes = [Pass(run_id=i, choice=choice_by_id(i)) for i in selected_ids]
        result = run_annotation(
            text=doc_text,
            labels=entity_labels,
            method_id=method_id,
            max_sentences=max_sentences,
            llm_params={"temperature": 0.0, "max_tokens": 500, "timeout": 45, "max_retries": 1},
            progress_callback=lambda i, n, s: status.write(f"Sentence {i + 1}/{n}"),
            estimator="model_agreement",
            passes=passes,
        )
    except Exception as exc:  # noqa: BLE001 — surface the error instead of crashing the page
        status.update(label="Failed", state="error")
        st.error(f"Comparison failed: {exc}")
        st.stop()

    status.update(label=f"Done — {result['n_llm_calls']} LLM calls", state="complete")
    st.session_state.cmp_result = result
    st.session_state.cmp_text = doc_text

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
result = st.session_state.get("cmp_result")
if not result:
    st.info("Pick two models and run a comparison to see the results.", icon=":material/info:")
    st.stop()

runs = [(pid, result["per_pass_entities"][pid]) for pid in result["pass_ids"]]
report = agreement_report(runs)

st.subheader(":material/insights: Agreement", anchor=False)
c1, c2, c3 = st.columns(3)
c1.metric("Distinct spans found (union)", report["n_union"])
c2.metric("Found by every model", report["n_unanimous"])
c3.metric(
    "Unanimous share",
    f"{report['share_unanimous']:.0%}",
    help="Share of all spans that every model agreed on. The rest are where a reviewer should look.",
)

if report["pairs"]:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Model A": p["a"], "Model B": p["b"],
                    "A found": p["n_a"], "B found": p["n_b"],
                    "Shared": p["shared"], "Only A": p["only_a"], "Only B": p["only_b"],
                    "Overlap (Jaccard)": round(p["jaccard"], 3),
                }
                for p in report["pairs"]
            ]
        ),
        hide_index=True,
        width="stretch",
    )

# --- Span-by-span table ----------------------------------------------------
st.subheader(":material/table_rows: Span by span", anchor=False)
st.caption("One row per distinct span. A tick means that model found it. Rows with a gap are the disagreements.")

sets = {pid: {span_key(e) for e in ents} for pid, ents in runs}
by_key = {}
for _pid, ents in runs:
    for e in ents:
        by_key.setdefault(span_key(e), e)

rows = []
for key, entity in sorted(by_key.items(), key=lambda kv: kv[0][0]):
    row = {"Text": entity["text"], "Label": entity["type"], "Start": key[0], "End": key[1]}
    found_count = 0
    for pid in result["pass_ids"]:
        hit = key in sets[pid]
        row[pid] = hit
        found_count += int(hit)
    row["Agreement"] = found_count / len(result["pass_ids"])
    rows.append(row)

df = pd.DataFrame(rows).sort_values("Agreement", kind="stable").reset_index(drop=True)
st.dataframe(
    df,
    hide_index=True,
    width="stretch",
    column_config={
        **{pid: st.column_config.CheckboxColumn(pid, disabled=True) for pid in result["pass_ids"]},
        "Agreement": st.column_config.ProgressColumn("Agreement", min_value=0.0, max_value=1.0, format="%.2f"),
    },
)

disagreements = [r for r in rows if 0 < r["Agreement"] < 1]
if disagreements:
    st.warning(
        f"**{len(disagreements)} spans** were found by some models but not others. Those are the ones "
        "worth a human's time, and they are what the *Model agreement* confidence option on the "
        "Annotate page scores.",
        icon=":material/priority_high:",
    )
else:
    st.success("The models agreed on every span in this text.", icon=":material/check_circle:")
