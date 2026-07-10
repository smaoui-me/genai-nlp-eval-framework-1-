"""
The Extraction Method Evaluation page: charts comparing how well each
LLM pre-labeling method did on a benchmark test set. This page never calls
the LLM — it only reads and charts CSV files that were already computed
(see utils/evaluation_data.py for where the numbers come from).

Charting library: Plotly Express (imported as `px`). We build a chart with
`px.bar(...)`, tweak its layout a little, then show it with
`st.plotly_chart(...)`.
"""

import plotly.express as px
import streamlit as st

from utils.annotation_store import init_session_state
from utils.evaluation_data import LEGACY_METHODS, is_available, load_errors, load_per_type, load_scores

init_session_state()

st.title(":material/query_stats: Extraction method evaluation")
st.caption(
    "Benchmark results for each LLM pre-labeling method, from the "
    "[genai-nlp-eval-framework-1-](https://github.com/smaoui-me/genai-nlp-eval-framework-1-) project's "
    "runs on the FewNERD dataset"
)

if not is_available():
    st.warning(
        "No benchmark score files found. They should be bundled in this repo under "
        "`data/eval_benchmarks/` — check that folder wasn't deleted or excluded from the deployment.",
        icon=":material/warning:",
    )
    st.stop()

all_scores = load_scores()  # includes the 2 legacy runs — used only by the per-entity-type breakdown below
scores = load_scores(include_legacy=False)  # the 4 real Annotate methods — used by every chart above that
if scores.empty:
    st.info("No evaluation CSVs found yet.", icon=":material/info:")
    st.stop()

st.subheader(":material/leaderboard: Method comparison", anchor=False)

metric_choice = st.radio(
    "Matching mode", ["strict", "lenient"], horizontal=True,
    help="Strict = exact text + span match. Lenient = text match only (ignores span boundaries).",
)
prf_cols = [f"{metric_choice}_precision", f"{metric_choice}_recall", f"{metric_choice}_f1"]

# scores has one row per method and one column per metric. Plotly's grouped
# bar chart wants "long" data instead: one row per (method, metric, score).
# pandas' .melt() reshapes it that way — a common step before charting.
melted = scores.melt(
    id_vars=["display_name"], value_vars=prf_cols, var_name="metric", value_name="score"
)
melted["metric"] = melted["metric"].str.replace(f"{metric_choice}_", "", regex=False).str.upper()

fig = px.bar(
    melted, x="display_name", y="score", color="metric", barmode="group",
    labels={"display_name": "Method", "score": "Score"}, range_y=[0, 1],
)
fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), legend_title_text="")
st.plotly_chart(fig, width="stretch")

st.subheader(":material/rule: Reliability", anchor=False)
col1, col2 = st.columns(2)
with col1:
    # The chart title is a plain st.caption above the chart, not a Plotly
    # `title=`, so it always has room to fully display instead of getting
    # clipped by the chart's small top margin.
    st.caption("Output was valid, parseable JSON")
    fig_json = px.bar(
        scores, x="display_name", y="json_valid_rate",
        labels={"display_name": "Method", "json_valid_rate": "JSON valid rate"}, range_y=[0, 1],
    )
    fig_json.update_layout(margin=dict(t=10, l=0, r=0, b=0))
    st.plotly_chart(fig_json, width="stretch")
with col2:
    st.caption("Examples with an out-of-schema label/entity")
    fig_invalid = px.bar(
        scores, x="display_name", y="invalid_label_rate",
        labels={"display_name": "Method", "invalid_label_rate": "Invalid label rate"}, range_y=[0, 1],
    )
    fig_invalid.update_layout(margin=dict(t=10, l=0, r=0, b=0))
    st.plotly_chart(fig_invalid, width="stretch")

with st.expander("Raw scores table"):
    st.dataframe(
        scores.set_index("display_name")[
            ["model", "n_examples"] + [c for c in scores.columns if c.endswith(("precision", "recall", "f1", "rate"))]
        ],
        width="stretch",
    )

st.subheader(":material/category: Per-entity-type breakdown", anchor=False)
st.caption(
    "How well a method did on each specific entity type (e.g. `location-GPE` vs. `organization-company`), "
    "instead of one overall score. Only the two **legacy** multi-label runs have this — the four Annotate "
    "methods only ever predict a single type (`location`), so there's nothing to break down by type. The "
    "legacy runs aren't an Annotate option; they're kept here just for this breakdown."
)
legacy_scores = all_scores[all_scores["method"].isin(LEGACY_METHODS)]
method_for_breakdown = st.selectbox(
    "Method", legacy_scores["method"].tolist(),
    format_func=lambda m: legacy_scores.set_index("method").loc[m, "display_name"],
    key="breakdown_method",
)
per_type = load_per_type(method_for_breakdown)
if per_type is not None and not per_type.empty:
    fig_type = px.bar(
        per_type.head(15), x="entity_type", y=["precision", "recall", "f1"], barmode="group", range_y=[0, 1],
    )
    fig_type.update_layout(margin=dict(t=10, l=0, r=0, b=0), legend_title_text="", xaxis_title="")
    st.plotly_chart(fig_type, width="stretch")
else:
    st.caption("No per-type breakdown available for this method.")

errors = load_errors(method_for_breakdown)
if errors is not None and not errors.empty:
    with st.expander(":material/bug_report: Lowest-scoring examples (for intuition on failure modes)"):
        st.dataframe(errors[["text", "gold_spans", "pred_spans", "strict_f1"]], width="stretch", height=300)
