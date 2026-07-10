"""
The entry point — the file you run with `streamlit run app.py`. It only
sets up the page and hands off to whichever page the user picked; the
actual page content lives in the pages/ folder.

This is Streamlit's standard multi-page pattern: one small "router" file
(this one) that lists the pages, and one file per page with that page's code.
"""

import streamlit as st

from utils.annotation_store import init_session_state
from utils.labels import DEFAULT_LABELS

# Controls the browser tab: title, icon, and full-width ("wide") layout.
# Must be the very first Streamlit command in the file.
st.set_page_config(page_title="NLP Annotation Tool", page_icon=":material/edit_note:", layout="wide")

# Set up st.session_state (see utils/annotation_store.py) before any page
# runs, so every page can rely on these values already existing.
init_session_state()
if st.session_state.labels is None:
    st.session_state.labels = list(DEFAULT_LABELS)

# st.Page describes one page: which file holds it, its title, its icon.
page_home = st.Page("pages/0_home.py", title="Home", icon=":material/home:", default=True)
page_annotate = st.Page("pages/1_annotate.py", title="Annotate", icon=":material/edit_note:")
page_extraction_eval = st.Page(
    "pages/2_extraction_evaluation.py", title="Extraction Method Evaluation", icon=":material/query_stats:"
)
page_annotation_eval = st.Page(
    "pages/3_annotation_evaluation.py", title="Annotation Evaluation", icon=":material/fact_check:"
)

# Builds the sidebar menu, grouped under these section headers ("" = no header).
pg = st.navigation(
    {
        "": [page_home],
        "Workflow": [page_annotate],
        "Evaluation": [page_extraction_eval, page_annotation_eval],
    }
)

pg.run()  # runs the code for whichever page is currently selected
