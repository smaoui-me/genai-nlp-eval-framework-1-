"""
The Annotate page — the main tool. Three steps, top to bottom:

    1. Configure  - pick a document, entity labels, and a pre-labeling method
    2. Review     - see the LLM's suggested entities, fix/confirm/delete them
    3. Export     - save the final result as gold-standard JSON

Streamlit tip: every widget below has a `key="..."` so Streamlit can tell
widgets apart and remember their values between reruns (the whole script
re-runs top to bottom on every click — see utils/annotation_store.py for
more on st.session_state, which is how values survive a rerun).
"""

import json

import streamlit as st

from utils.annotation_store import (
    add_manual_entity,
    apply_edits,
    bump_entities_version,
    build_gold_export,
    confirm_all_pending,
    entities_to_dataframe,
    init_session_state,
    new_entity_id,
    render_highlighted_html,
    save_gold_export,
)
from utils.extraction_methods import METHODS, run_annotation
from utils.file_utils import extract_text
from utils.labels import DEFAULT_LABELS
from utils.llm_client import get_model_name, is_configured, missing_credentials_message
from utils.sample_data import SAMPLES

# A safety limit on how much text we'll try to annotate at once, so one
# huge paste/upload can't make the app slow or run up a huge LLM bill.
MAX_CHARS = 20_000

init_session_state()
if st.session_state.labels is None:
    st.session_state.labels = list(DEFAULT_LABELS)
if st.session_state.available_labels is None:
    st.session_state.available_labels = set(DEFAULT_LABELS) | {"art", "other"}

st.title(":material/edit_note: Annotate")
st.caption("Raw text → LLM pre-annotation → human correction → gold-standard JSON")

# ---------------------------------------------------------------------------
# Step 1: Configure — choose a document, labels, method, and model
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.subheader(":material/tune: 1 · Configure", anchor=False)

    col_source, col_doc = st.columns([1, 2])
    with col_source:
        source_mode = st.radio(
            "Document source", ["Sample document", "Upload file", "Paste text"], key="source_mode"
        )
    with col_doc:
        if source_mode == "Sample document":
            sample_name = st.selectbox("Sample", list(SAMPLES.keys()), key="sample_name")
            doc_text = SAMPLES[sample_name]
            doc_name = sample_name
        elif source_mode == "Upload file":
            uploaded = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"], key="uploaded_file")
            if uploaded is not None:
                doc_text = extract_text(uploaded)
                doc_name = uploaded.name
            else:
                doc_text, doc_name = "", None
        else:  # "Paste text"
            # max_chars stops the browser from even accepting more input
            # than this, so we don't need to truncate anything ourselves.
            doc_text = st.text_area(
                "Paste text to annotate", height=180, max_chars=MAX_CHARS, key="pasted_text",
                placeholder="Paste a document, email, or ticket here...",
            )
            doc_name = "Pasted text"
            st.caption(f"{len(doc_text)} / {MAX_CHARS} characters")

    if doc_text:
        with st.expander(f"Preview raw text ({len(doc_text)} characters)"):
            st.text(doc_text[:3000] + ("..." if len(doc_text) > 3000 else ""))

    col_labels, col_method = st.columns([2, 2])
    with col_labels:
        st.session_state.labels = st.multiselect(
            "Entity labels to annotate",
            options=sorted(st.session_state.available_labels),
            default=st.session_state.labels,
            key="labels_select",
            help="Deselecting a label just hides it here — it stays available if you want it back later.",
        )
        new_label = st.text_input("Add a custom label", placeholder="e.g. product_id", key="new_label_input")
        if st.button(":material/add: Add label", key="add_label_btn") and new_label.strip():
            label = new_label.strip().lower()
            st.session_state.available_labels.add(label)
            if label not in st.session_state.labels:
                st.session_state.labels.append(label)
            st.rerun()  # restart the script so the new label shows up in the picker right away

    with col_method:
        method_id = st.selectbox(
            "Annotation method (LLM pre-labeling)",
            options=list(METHODS.keys()),
            format_func=lambda m: METHODS[m]["label"],  # show a friendly label, keep the short ID as the real value
            key="method_id",
        )
        st.caption(METHODS[method_id]["description"])
        with st.expander("Advanced"):
            max_sentences = st.slider("Max sentences to process", 1, 25, 6, key="max_sentences")
            temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1, key="temperature")

            # Model picker: defaults to whatever is configured in secrets
            # (always works), but lets you try a different model/deployment
            # name too, e.g. a smaller/cheaper one, if your LLM gateway has
            # one available. accept_new_options=True lets you type a name
            # that isn't in the preset list below.
            default_model = get_model_name()
            model_presets = [m for m in [default_model, "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano"] if m]
            selected_model = st.selectbox(
                "Model",
                options=model_presets,
                accept_new_options=True,
                key="selected_model",
                help=(
                    f"Defaults to **{default_model}** (from your credentials), which is guaranteed to work. "
                    "The other options are just common model names — whether they actually work depends on "
                    "what your LLM gateway has deployed."
                ),
            )

    run_disabled = not doc_text or not st.session_state.labels or not is_configured()
    if not is_configured():
        st.warning(missing_credentials_message(), icon=":material/warning:")
    run_clicked = st.button(
        ":material/auto_awesome: Run LLM pre-annotation", type="primary", disabled=run_disabled, width="stretch",
        key="run_button",
    )

if run_clicked:
    status_box = st.status("Running extraction...", expanded=True)

    def _progress(i, n, sentence_text):
        status_box.write(f"Sentence {i + 1}/{n}: _{sentence_text[:70]}{'...' if len(sentence_text) > 70 else ''}_")

    try:
        result = run_annotation(
            text=doc_text,
            labels=st.session_state.labels,
            method_id=method_id,
            max_sentences=max_sentences,
            llm_params={
                "temperature": temperature, "max_tokens": 500, "timeout": 45, "max_retries": 1,
                "model": selected_model,
            },
            progress_callback=_progress,
        )
    except Exception as exc:  # noqa: BLE001 - show any LLM/config error to the user instead of crashing
        status_box.update(label="Failed", state="error")
        st.error(f"Extraction failed: {exc}")
        st.stop()

    status_box.update(label=f"Done — {len(result['entities'])} entities found", state="complete")

    entities = [
        {
            "id": new_entity_id(), "text": e["text"], "type": e["type"], "start": e["start"], "end": e["end"],
            "source": "model", "model_name": selected_model, "status": "pending",
        }
        for e in result["entities"]
    ]
    st.session_state.entities = entities
    st.session_state.doc_text = doc_text
    st.session_state.doc_name = doc_name
    st.session_state.last_run_meta = {
        "method": method_id,
        "n_sentences": result["n_sentences"],
        "n_sentences_total": result["n_sentences_total"],
    }
    bump_entities_version()  # forces the review table below to redraw with this fresh data

# ---------------------------------------------------------------------------
# Step 2: Review — see the suggested entities and correct them
# ---------------------------------------------------------------------------
if st.session_state.doc_text:
    st.subheader(":material/visibility: 2 · Review", anchor=False)
    meta = st.session_state.last_run_meta or {}
    if meta.get("n_sentences_total", 0) > meta.get("n_sentences", 0):
        st.caption(
            f"Processed {meta['n_sentences']} of {meta['n_sentences_total']} sentences "
            "(raise the sentence limit above to cover the full document)."
        )

    entities = st.session_state.entities
    labels = st.session_state.labels

    st.markdown(render_highlighted_html(st.session_state.doc_text, entities, labels), unsafe_allow_html=True)

    visible = [e for e in entities if e["status"] != "deleted"]
    pending = sum(1 for e in visible if e["status"] == "pending")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Entities", len(visible))
    m2.metric("Pending review", pending)
    m3.metric("Confirmed", sum(1 for e in visible if e["status"] == "confirmed"))
    m4.metric("Relabeled", sum(1 for e in visible if e["status"] == "edited"))

    st.caption(
        "Tick **confirmed** once you've checked a row, change **label** to relabel it, or tick "
        "**delete** to remove it — then click **Save corrections**."
    )
    # The key includes entities_version so that whenever we change the
    # entities from a button (not by editing this table), Streamlit treats
    # it as a brand-new table and redraws it with the fresh data, instead of
    # keeping whatever it last showed on screen.
    editor_key = f"entity_editor_{st.session_state.entities_version}"
    edited_df = st.data_editor(
        entities_to_dataframe(entities),
        column_config={
            "id": None,  # hides this column — it's only used internally to match rows back up
            "text": st.column_config.TextColumn("Text", disabled=True),
            "type": st.column_config.SelectboxColumn("Label", options=labels),
            "start": st.column_config.NumberColumn("Start", disabled=True),
            "end": st.column_config.NumberColumn("End", disabled=True),
            "source": st.column_config.TextColumn("Source", disabled=True, help="Which model suggested this, or 'human' if you added it"),
            "status": st.column_config.TextColumn("Status", disabled=True),
            "confirmed": st.column_config.CheckboxColumn("Confirmed"),
            "delete": st.column_config.CheckboxColumn("Delete"),
        },
        hide_index=True,
        width="stretch",
        key=editor_key,
    )

    col_save, col_confirm, col_reset = st.columns(3)
    if col_save.button(":material/save: Save corrections", width="stretch", key="save_corrections_btn"):
        st.session_state.entities = apply_edits(entities, edited_df, labels)
        bump_entities_version()
        st.rerun()
    if col_confirm.button(":material/done_all: Confirm all pending", width="stretch", key="confirm_all_btn"):
        n_before = sum(1 for e in entities if e["status"] == "pending")
        st.session_state.entities = confirm_all_pending(entities)
        bump_entities_version()
        st.toast(f"Confirmed {n_before} pending entities", icon=":material/done_all:")
        st.rerun()
    if col_reset.button(":material/refresh: Clear annotation", width="stretch", key="clear_btn"):
        st.session_state.entities = []
        st.session_state.doc_text = ""
        st.session_state.doc_name = None
        bump_entities_version()
        st.rerun()

    with st.expander(":material/add_circle: Add an entity the model missed"):
        add_col1, add_col2, add_col3 = st.columns([2, 1, 1])
        manual_text = add_col1.text_input("Exact text (must match the document)", key="manual_text")
        manual_type = add_col2.selectbox("Label", labels, key="manual_type")
        if add_col3.button("Add", key="manual_add_btn", width="stretch") and manual_text.strip():
            new_entities, error = add_manual_entity(entities, manual_text.strip(), manual_type, st.session_state.doc_text)
            if error:
                st.error(error)
            else:
                st.session_state.entities = new_entities
                bump_entities_version()
                st.rerun()

    # -----------------------------------------------------------------
    # Step 3: Export — download or save the final reviewed result
    # -----------------------------------------------------------------
    st.subheader(":material/download: 3 · Export gold JSON", anchor=False)
    export = build_gold_export(
        doc_name=st.session_state.doc_name or "document",
        doc_text=st.session_state.doc_text,
        labels=labels,
        method_id=st.session_state.get("method_id", "few_shot_structured"),
        entities=entities,
    )

    col_dl, col_persist = st.columns(2)
    col_dl.download_button(
        ":material/download: Download gold_dataset.json",
        data=json.dumps(export, indent=2, ensure_ascii=False),
        file_name=f"{(st.session_state.doc_name or 'document').rsplit('.', 1)[0]}_gold.json",
        mime="application/json",
        width="stretch",
        key="download_gold_btn",
    )
    if col_persist.button(":material/save: Save to project (for Annotation Evaluation)", width="stretch", key="persist_gold_btn"):
        path = save_gold_export(export)
        st.success(f"Saved {path.name}", icon=":material/check_circle:")
else:
    st.info("Pick a document above and run pre-annotation to get started.", icon=":material/info:")
