"""
The Annotation Evaluation page grades model suggestions against *your own*
review history from the Annotate page. Every time someone reviews a
document and clicks "Save to project", the full review trail gets saved as
a JSON file (see save_gold_export() in utils/annotation_store.py). This
page reads those files back in and turns them into charts.

Counter (from Python's collections module) counts how many times each
distinct thing appears — e.g. "how many times was a location relabeled?".
"""

from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.annotation_store import (
    compute_annotation_metrics,
    compute_confidence_diagnostics,
    init_session_state,
    load_gold_exports,
)
from utils.extraction_methods import METHODS
from utils.scirex_evaluation import evaluate_scirex_export

init_session_state()

st.title(":material/fact_check: Annotation evaluation")
st.caption("How much did LLM pre-labeling actually save the human reviewer?")

exports = load_gold_exports()

if not exports:
    st.info(
        "No saved annotations yet. Go to **Annotate**, run pre-annotation on a document, review it, "
        "and click **Save to project** to start building this evaluation.",
        icon=":material/info:",
    )
    st.stop()

# Turn each saved export into one row of summary numbers (see
# compute_annotation_metrics()), then combine into one table.
rows = []
for export in exports:
    metrics = compute_annotation_metrics(export)
    # **metrics spreads its key/value pairs into this new dict, alongside doc_name etc.
    rows.append({"doc_name": export["doc_name"], "method": export["method"], "exported_at": export["exported_at"], **metrics})
summary_df = pd.DataFrame(rows)

# --- Top summary numbers ---
st.subheader(":material/summarize: Overview", anchor=False)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Reviewed documents", len(exports))
# .notna().any() checks there's at least one real value, so we show "—"
# instead of a misleading 0% when there's no data yet.
m2.metric("Avg. acceptance rate", f"{summary_df['acceptance_rate'].mean():.0%}" if summary_df["acceptance_rate"].notna().any() else "—")
m3.metric("Avg. edit rate", f"{summary_df['edit_rate'].mean():.0%}" if summary_df["edit_rate"].notna().any() else "—")
m4.metric("Avg. deletion rate", f"{summary_df['deletion_rate'].mean():.0%}" if summary_df["deletion_rate"].notna().any() else "—")

st.caption(
    "**Acceptance rate** = model-suggested entities kept unchanged. **Edit rate** = relabeled by the human. "
    "**Deletion rate** = removed by the human as wrong. Higher acceptance / lower edit+deletion means the "
    "pre-labeling method needs less manual correction."
)

# --- Optional department-routing review -----------------------------------
routing_rows = []
for export in exports:
    classification = export.get("classification") or {}
    if classification.get("task") == "department_routing":
        routing_rows.append({
            "document": export["doc_name"],
            "classifier": classification.get("classifier", "llm"),
            "model_prediction": classification.get("model_prediction"),
            "approved_department": classification.get("approved_department"),
            "review_status": classification.get("review_status", "unknown"),
            "confidence": classification.get("confidence"),
            "evidence": classification.get("evidence", ""),
        })

st.subheader(":material/account_tree: Department-routing review", anchor=False)
if routing_rows:
    routing_df = pd.DataFrame(routing_rows)
    usable = routing_df[routing_df["review_status"].isin(["confirmed", "corrected"])]
    confirmed_routes = int((usable["review_status"] == "confirmed").sum())
    route_col1, route_col2, route_col3 = st.columns(3)
    route_col1.metric("Reviewed routing decisions", len(usable))
    route_col2.metric(
        "Accepted unchanged",
        f"{confirmed_routes / len(usable):.0%}" if len(usable) else "—",
    )
    route_col3.metric("Corrected by reviewer", int((usable["review_status"] == "corrected").sum()))
    st.caption(
        "This is workflow acceptance, not independent classification accuracy. For accuracy, compare "
        "predictions with departments assigned before reviewers see the model suggestion."
    )
    by_classifier = (
        usable.assign(accepted=usable["review_status"].eq("confirmed"))
        .groupby("classifier", as_index=False)
        .agg(reviewed=("document", "count"), accepted=("accepted", "sum"))
    )
    by_classifier["acceptance_rate"] = (
        100 * by_classifier["accepted"] / by_classifier["reviewed"]
    )
    st.markdown("##### Routing review by classifier")
    st.dataframe(
        by_classifier,
        hide_index=True,
        width="stretch",
        column_config={
            "acceptance_rate": st.column_config.NumberColumn(format="%.1f%%")
        },
    )
    st.dataframe(routing_df, hide_index=True, width="stretch")
else:
    st.caption("No reviewed department-routing suggestions have been saved yet.")

# --- Direct SciREX gold comparison -----------------------------------------
st.subheader(":material/analytics: SciREX prediction accuracy", anchor=False)
scirex_rows = []
scirex_unavailable = []
for export in exports:
    evaluation = evaluate_scirex_export(export)
    if evaluation["available"]:
        strict = evaluation["strict"]
        scirex_rows.append({
            "document": export["doc_name"], "method": export["method"],
            "precision": strict["precision"], "recall": strict["recall"], "f1": strict["f1"],
            "tp": strict["tp"], "fp": strict["fp"], "fn": strict["fn"],
            "predicted": evaluation["predicted_entities"], "gold": evaluation["gold_entities"],
            "scope": evaluation["scope"],
            "tolerant_tp": evaluation["boundary_tolerant"]["tp"],
            "tolerant_fp": evaluation["boundary_tolerant"]["fp"],
            "tolerant_fn": evaluation["boundary_tolerant"]["fn"],
            "overlap_tp": evaluation["overlap"]["tp"],
            "overlap_fp": evaluation["overlap"]["fp"],
            "overlap_fn": evaluation["overlap"]["fn"],
        })
    elif (export.get("source") or {}).get("source_dataset") == "scirex":
        scirex_unavailable.append(f"{export['doc_name']}: {evaluation['reason']}")

if scirex_rows:
    scirex_df = pd.DataFrame(scirex_rows)
    total_tp, total_fp, total_fn = scirex_df[["tp", "fp", "fn"]].sum()
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall else 0.0
    )
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Strict precision", f"{micro_precision:.1%}")
    a2.metric("Strict recall", f"{micro_recall:.1%}")
    a3.metric("Strict F1", f"{micro_f1:.1%}")
    a4.metric("Evaluated runs", len(scirex_df))
    def aggregate_f1(prefix):
        tp = scirex_df[f"{prefix}_tp"].sum()
        fp = scirex_df[f"{prefix}_fp"].sum()
        fn = scirex_df[f"{prefix}_fn"].sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f1_col1, f1_col2 = st.columns(2)
    f1_col1.metric("±1/±2 token boundary F1", f"{aggregate_f1('tolerant'):.1%}")
    f1_col2.metric("Same-label overlap F1", f"{aggregate_f1('overlap'):.1%}")
    st.caption(
        "Exact character span + label matching against hidden SciREX gold. Only the text prefix "
        "actually sent to the LLM is scored. Human corrections are not used as predictions. "
        "Tolerant matching permits one token at the start and two at the end; overlap matching "
        "measures whether a reviewer was directed to the right labeled phrase."
    )
    st.dataframe(scirex_df, hide_index=True, width="stretch")
else:
    st.caption("No saved SciREX pre-annotation run is available for direct gold scoring yet.")
if scirex_unavailable:
    st.warning("Some SciREX exports could not be scored:\n\n- " + "\n- ".join(scirex_unavailable))

# --- Chart: acceptance/edit/deletion rate, one bar per document ---
st.subheader(":material/bar_chart: Per-document breakdown", anchor=False)
# Reshape from 3 separate columns into one "metric" + "rate" column pair —
# the shape Plotly wants for a grouped/stacked bar chart.
melted = summary_df.melt(
    id_vars=["doc_name", "method"],
    value_vars=["acceptance_rate", "edit_rate", "deletion_rate"],
    var_name="metric",
    value_name="rate",
)
melted["metric"] = melted["metric"].str.replace("_rate", "", regex=False)
fig = px.bar(melted, x="doc_name", y="rate", color="metric", barmode="stack", range_y=[0, 1.05], labels={"doc_name": "Document"})
fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), legend_title_text="")
st.plotly_chart(fig, width="stretch")

# --- Chart: which labels get relabeled/deleted most often ---
st.subheader(":material/label: Where humans intervene, by label", anchor=False)
label_events = Counter()
for export in exports:
    for e in export.get("review_log", []):
        if e["source"] == "model" and e["status"] in ("edited", "deleted"):
            label_events[(e["type"], e["status"])] += 1  # Counter keys can be tuples, e.g. ("location", "edited")

if label_events:
    label_df = pd.DataFrame(
        [{"type": t, "action": a, "count": c} for (t, a), c in label_events.items()]
    )
    fig_label = px.bar(label_df, x="type", y="count", color="action", barmode="group")
    fig_label.update_layout(margin=dict(t=10, l=0, r=0, b=0), legend_title_text="")
    st.plotly_chart(fig_label, width="stretch")
else:
    st.caption("No relabels or deletions logged yet — every model suggestion has been accepted as-is so far.")

# --- Was the confidence score worth the calls it cost? ---------------------
# This is the honest test of the uncertainty feature. A score only earns its
# keep if the rows it flags are the ones the reviewer actually changes.
st.subheader(":material/target: Did the confidence score point at the right rows?", anchor=False)

scored_spans = [
    e
    for export in exports
    for e in export.get("review_log", [])
    if e.get("source") == "model" and isinstance(e.get("confidence"), (int, float))
]

if not scored_spans:
    st.caption(
        "No reviewed document carries confidence scores yet. Run a pre-annotation with a "
        "confidence estimator turned on, review it, and save it here."
    )
else:
    diag = compute_confidence_diagnostics(scored_spans)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Spans with a score", diag["n_scored"])
    d2.metric(
        "Changed when flagged",
        f"{diag['change_rate_flagged']:.0%}" if diag["change_rate_flagged"] is not None else "—",
        help="Share of low-confidence spans the reviewer edited or deleted.",
    )
    d3.metric(
        "Changed when confident",
        f"{diag['change_rate_unflagged']:.0%}" if diag["change_rate_unflagged"] is not None else "—",
        help="Same, for the spans the model was sure about.",
    )
    d4.metric(
        "Lift",
        "∞" if diag["lift"] == float("inf") else (f"{diag['lift']:.2f}x" if diag["lift"] else "—"),
        help="How much more often a flagged span was changed. 1.0x means the score told you nothing.",
    )
    if diag["lift"] and diag["lift"] != float("inf") and diag["lift"] < 1.2:
        st.warning(
            "Flagged spans were changed about as often as confident ones, so on this data the "
            "score is not helping. Either the estimator has no signal here, or there are too few "
            "reviewed spans to tell yet.",
            icon=":material/warning:",
        )

with st.expander(":material/table_rows: Raw per-document metrics"):
    st.dataframe(summary_df, width="stretch", hide_index=True)

with st.expander(":material/folder: Reviewed documents"):
    for export in exports:
        method_label = METHODS.get(export["method"], {}).get("label", export["method"])
        st.markdown(f"**{export['doc_name']}** · {method_label} · {len(export.get('gold_entities', []))} gold entities")
