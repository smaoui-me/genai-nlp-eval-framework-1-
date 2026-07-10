"""
The Home page: a quick overview of what the app does, plus a small always-
visible preview of what the review screen looks like (using fake, hardcoded
data — this page never calls the LLM).

Streamlit runs this whole file top to bottom every time someone opens or
interacts with this page — there's no special "main function" to look for.
"""

import streamlit as st

from utils.annotation_store import init_session_state, render_highlighted_html
from utils.llm_client import get_model_name, is_configured, missing_credentials_message

init_session_state()

# --- Fake example data for the preview below. Not connected to the real app. ---
LABELS_DEMO = ["location", "organization", "person"]
DEMO_TEXT = (
    "Customer reported a delayed delivery in Munich after shipment from DHL Hub. "
    "Please contact John Miller before Friday and verify the destination address in Berlin."
)


def _demo_entities() -> list[dict]:
    """Build a fake list of entities (same shape as the real ones) for the preview."""
    spans = [("Munich", "location", "confirmed"), ("DHL Hub", "organization", "confirmed"),
             ("John Miller", "person", "pending"), ("Berlin", "location", "confirmed")]
    entities = []
    cursor = 0
    for text, etype, status in spans:
        start = DEMO_TEXT.index(text, cursor)
        end = start + len(text)
        cursor = end
        entities.append({"id": text, "text": text, "type": etype, "start": start, "end": end,
                          "source": "model", "status": status})
    return entities


st.title(":material/edit_note: Annotation tool")
st.caption("Create gold-standard NER data with LLM pre-labeling + human review")

if is_configured():
    st.success(f"LLM connected — model **{get_model_name()}**", icon=":material/check_circle:")
else:
    st.warning("No LLM credentials found. " + missing_credentials_message(), icon=":material/warning:")

# --- The 4-step flow, one card per step ---
st.subheader(":material/route: Human-in-the-loop flow", anchor=False)

flow = st.columns(4)
steps = [
    ("1", "description", "Raw text", "An unlabeled document or ticket"),
    ("2", "auto_awesome", "LLM pre-labels", "Suggests entities (source: model)"),
    ("3", "fact_check", "Human review", "Confirm, relabel, or delete each one"),
    ("4", "task_alt", "Gold export", "Final reviewed JSON dataset"),
]
for col, (num, icon, title, sub) in zip(flow, steps):
    with col, st.container(border=True):
        st.badge(f"Step {num}", color="orange")
        st.markdown(f"**:material/{icon}: {title}**", text_alignment="center")
        st.caption(sub, text_alignment="center")

st.write("")
st.caption(
    "**Model suggests entities** → **human confirms** → **fixes a label or boundary** → "
    "**deletes false positives** → **exports gold JSON**"
)
st.caption(
    "**Gold JSON** is the final, human-reviewed file this tool produces — the trustworthy "
    "dataset used to evaluate or train other models."
)

# --- Live preview of the highlighted-text review UI, using fake data above ---
st.subheader(":material/visibility: What the review screen looks like", anchor=False)
# unsafe_allow_html=True is needed because render_highlighted_html() returns
# a string of HTML — Streamlit escapes HTML by default, so we opt in here.
st.markdown(render_highlighted_html(DEMO_TEXT, _demo_entities(), LABELS_DEMO), unsafe_allow_html=True)
st.caption("Solid border = confirmed by a human · dashed border = still pending review. Try it live on the **Annotate** page.")

# --- Three short callouts pointing to the other pages ---
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.subheader(":material/edit_note: Start annotating", anchor=False)
    st.write(
        "Upload a document, pick your entity labels, and choose a pre-labeling method — "
        "zero-shot or few-shot, freeform or structured JSON output — on the "
        "**Annotate** page."
    )
with col_b:
    st.subheader(":material/query_stats: Which method is best?", anchor=False)
    st.write(
        "Each pre-labeling strategy was benchmarked on FewNERD. See precision/recall/F1 "
        "and JSON validity per method on the **Extraction Method Evaluation** page."
    )
with col_c:
    st.subheader(":material/fact_check: How much does the LLM get right?", anchor=False)
    st.write(
        "Every human correction (confirm / relabel / delete / add) is logged. The "
        "**Annotation Evaluation** page turns that review trail into acceptance and edit "
        "rates — a measure of how much manual work the pre-labeling actually saves."
    )
