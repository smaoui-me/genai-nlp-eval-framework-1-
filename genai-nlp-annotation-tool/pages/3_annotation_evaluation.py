"""
The Annotation Evaluation page: unlike the previous page (which grades the
LLM against a research benchmark), this one grades it against *your own*
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

from utils.annotation_store import compute_annotation_metrics, init_session_state, load_gold_exports
from utils.extraction_methods import METHODS

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

with st.expander(":material/table_rows: Raw per-document metrics"):
    st.dataframe(summary_df, width="stretch", hide_index=True)

with st.expander(":material/folder: Reviewed documents"):
    for export in exports:
        method_label = METHODS.get(export["method"], {}).get("label", export["method"])
        st.markdown(f"**{export['doc_name']}** · {method_label} · {len(export.get('gold_entities', []))} gold entities")
