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

import io
import json

import pandas as pd
import streamlit as st

from utils.annotation_store import (
    add_manual_entity,
    apply_edits,
    assign_span_numbers,
    bump_entities_version,
    build_gold_export,
    confirm_all_pending,
    entities_to_dataframe,
    init_session_state,
    new_entity_id,
    render_highlighted_html,
    save_gold_export,
)
from utils.benchmark_data import load_scirex_benchmark, source_metadata
from utils.extraction_methods import METHODS, build_passes, run_annotation
from utils.embedding_routing import (
    build_routing_index,
    embedding_routing_status,
    route_ticket_with_embeddings,
)
from utils.file_utils import extract_text
from utils.labels import DEFAULT_LABELS
from utils.llm_client import is_configured, missing_credentials_message
from utils.model_registry import available_choices, choice_by_id, default_choice
from utils.prompt_builder import prompt_hash, suggest_prompt
from utils.sample_data import SAMPLES
from utils.ticket_routing import route_ticket, suggest_routing_prompt
from utils.tokenizer import split_sentences
from utils.use_case_config import load_use_cases
from utils.uncertainty import (
    DEFAULT_REVIEW_BUDGET,
    DEFAULT_REVIEW_THRESHOLD,
    ESTIMATORS,
    FLAG_MODES,
    HIGH_UNCERTAINTY_BELOW,
    confidence_spread,
    has_signal,
    summarise,
)

# A safety limit on how much text we'll try to annotate at once, so one
# huge paste/upload can't make the app slow or run up a huge LLM bill.
MAX_CHARS = 20_000

init_session_state()
USE_CASES = load_use_cases()
if st.session_state.labels is None:
    st.session_state.labels = list(DEFAULT_LABELS)
if st.session_state.available_labels is None:
    st.session_state.available_labels = set(DEFAULT_LABELS) | {"art", "other"}


def _add_custom_label() -> None:
    """Add and select a label inside Streamlit's pre-rerun callback."""
    label = st.session_state.get("new_label_input", "").strip()
    if not label:
        return
    st.session_state.available_labels.add(label)
    if label not in st.session_state.labels:
        st.session_state.labels.append(label)
    selected = list(st.session_state.get("labels_select", []))
    if label not in selected:
        selected.append(label)
    st.session_state.labels_select = selected
    st.session_state.new_label_input = ""


def _reset_prompt(suggestion: str) -> None:
    st.session_state.prompt_template_editor = suggestion


def _clear_annotation() -> None:
    """Clear review state before Streamlit recreates its widget-backed keys."""
    st.session_state.entities = []
    st.session_state.doc_text = ""
    st.session_state.doc_name = None
    st.session_state.last_run_meta = None
    st.session_state.routing_result = None
    st.session_state.routing_approved_department = None
    st.session_state.focus_span_id = None
    bump_entities_version()


st.title(":material/edit_note: Annotate")
st.caption("Raw text → LLM pre-annotation → human correction → gold-standard JSON")

# ---------------------------------------------------------------------------
# Step 1: Configure — choose a document, labels, method, and model
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.subheader(":material/tune: 1 · Configure", anchor=False)

    use_case_id = st.selectbox(
        "Use-case template",
        options=list(USE_CASES),
        index=list(USE_CASES).index("ticket_support") if "ticket_support" in USE_CASES else 0,
        format_func=lambda key: USE_CASES[key]["name"],
        key="use_case_id",
        help=(
            "Templates provide reusable labels, definitions, routing options, and prompt guidance. "
            "Add YAML files under config/use_cases for another domain."
        ),
    )
    selected_use_case = USE_CASES[use_case_id]
    if st.session_state.get("active_use_case_id") != use_case_id:
        preset_labels = selected_use_case["entity_labels"]
        st.session_state.available_labels.update(preset_labels)
        st.session_state.labels = list(preset_labels)
        st.session_state.labels_select = list(preset_labels)
        routing_defaults = selected_use_case.get("routing", {})
        st.session_state.enable_routing = bool(routing_defaults.get("enabled_by_default", False))
        st.session_state.routing_departments_editor = "\n".join(routing_defaults.get("departments", []))
        st.session_state.active_use_case_id = use_case_id
    st.caption(selected_use_case.get("description", ""))

    col_source, col_doc = st.columns([1, 2])
    with col_source:
        scirex_examples = load_scirex_benchmark()
        source_options = ["Sample document", "Upload file", "Paste text"]
        if scirex_examples:
            source_options.insert(1, "SciREX fixture")
        source_mode = st.radio("Document source", source_options, key="source_mode")
    with col_doc:
        current_source_meta = {}
        if source_mode == "Sample document":
            sample_name = st.selectbox("Sample", list(SAMPLES.keys()), key="sample_name")
            doc_text = SAMPLES[sample_name]
            doc_name = sample_name
        elif source_mode == "SciREX fixture":
            by_id = {example["example_id"]: example for example in scirex_examples}
            example_id = st.selectbox(
                "SciREX example", list(by_id),
                format_func=lambda key: (
                    f"{by_id[key]['length_bucket']} · {by_id[key]['sentence_count']} sentences · "
                    f"{by_id[key]['doc_id'][:12]}"
                ), key="scirex_example_id",
            )
            selected_example = by_id[example_id]
            doc_text = selected_example["text"]
            doc_name = selected_example["example_id"]
            current_source_meta = source_metadata(selected_example)
            scirex_labels = {"Method", "Task", "Metric", "Dataset"}
            st.session_state.available_labels.discard("Material")
            st.session_state.available_labels.update(scirex_labels)
            if st.session_state.get("active_schema_source") != "scirex":
                st.session_state.labels = sorted(scirex_labels)
                st.session_state.labels_select = sorted(scirex_labels)
                st.session_state.active_schema_source = "scirex"
            st.caption(
                f"Source document `{selected_example['doc_id']}` · split `{selected_example['source_split']}` · "
                f"{selected_example['length_bucket']} bucket"
            )
        elif source_mode == "Upload file":
            uploaded = st.file_uploader(
                "Upload a .txt, .pdf, or .csv file", type=["txt", "pdf", "csv"], key="uploaded_file"
            )
            if uploaded is not None:
                if uploaded.name.lower().endswith(".csv"):
                    try:
                        ticket_frame = pd.read_csv(io.BytesIO(uploaded.getvalue()))
                    except Exception as exc:  # noqa: BLE001 - explain malformed user input instead of crashing
                        st.error(f"Could not read this CSV: {exc}")
                        ticket_frame = pd.DataFrame()
                    if ticket_frame.empty:
                        doc_text, doc_name = "", None
                        st.warning("The uploaded CSV has no readable ticket rows.")
                    else:
                        columns = list(ticket_frame.columns)
                        preferred_text = next(
                            (name for name in ("text", "ticket_text", "description", "body", "content")
                             if name in columns),
                            columns[0],
                        )
                        text_column = st.selectbox(
                            "Main ticket text column", columns, index=columns.index(preferred_text),
                            key="csv_text_column",
                        )
                        context_defaults = [
                            name for name in ("title", "subject") if name in columns and name != text_column
                        ]
                        context_columns = st.multiselect(
                            "Additional context columns", [name for name in columns if name != text_column],
                            default=context_defaults, key="csv_context_columns",
                        )
                        id_options = ["(row number)", *columns]
                        preferred_id = next(
                            (name for name in ("ticket_id", "id", "key") if name in columns), "(row number)"
                        )
                        id_column = st.selectbox(
                            "Ticket identifier column", id_options, index=id_options.index(preferred_id),
                            key="csv_id_column",
                        )
                        row_index = st.selectbox(
                            "Ticket to annotate", options=list(range(len(ticket_frame))),
                            format_func=lambda index: (
                                f"Row {index + 1}" if id_column == "(row number)"
                                else str(ticket_frame.iloc[index][id_column])
                            ), key="csv_row_index",
                        )
                        row = ticket_frame.iloc[int(row_index)]
                        parts = [
                            f"{column}: {row[column]}" for column in [*context_columns, text_column]
                            if pd.notna(row[column]) and str(row[column]).strip()
                        ]
                        doc_text = "\n\n".join(parts)
                        ticket_id = str(row_index + 1) if id_column == "(row number)" else str(row[id_column])
                        doc_name = f"{uploaded.name}__{ticket_id}"
                        current_source_meta = {
                            "source_dataset": "uploaded_csv",
                            "source_file": uploaded.name,
                            "source_row": int(row_index),
                            "ticket_id": ticket_id,
                            "text_column": text_column,
                            "context_columns": list(context_columns),
                        }
                else:
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

    if source_mode != "SciREX fixture" and st.session_state.get("active_schema_source") == "scirex":
        preset_labels = selected_use_case["entity_labels"]
        st.session_state.available_labels.update(preset_labels)
        st.session_state.labels = list(preset_labels)
        st.session_state.labels_select = list(preset_labels)
        st.session_state.active_schema_source = use_case_id

    if doc_text:
        with st.expander(f"Preview raw text ({len(doc_text)} characters)"):
            st.text(doc_text[:3000] + ("..." if len(doc_text) > 3000 else ""))

    col_labels, col_method = st.columns([2, 2])
    with col_labels:
        label_defaults = {} if "labels_select" in st.session_state else {"default": st.session_state.labels}
        st.session_state.labels = st.multiselect(
            "Entity labels to annotate",
            options=sorted(st.session_state.available_labels),
            key="labels_select",
            help="Deselecting a label just hides it here — it stays available if you want it back later.",
            **label_defaults,
        )
        new_label = st.text_input("Add a custom label", placeholder="e.g. product_id", key="new_label_input")
        st.button(
            ":material/add: Add label", key="add_label_btn", on_click=_add_custom_label,
            disabled=not new_label.strip(),
        )

    with col_method:
        method_id = st.selectbox(
            "Annotation method (LLM pre-labeling)",
            options=list(METHODS.keys()),
            format_func=lambda m: METHODS[m]["label"],  # show a friendly label, keep the short ID as the real value
            key="method_id",
        )
        st.caption(METHODS[method_id]["description"])

        # Model picker. Every configured endpoint from utils/model_registry.py
        # shows up here — add endpoints there (or in secrets.toml), not here.
        choices = available_choices()
        default = default_choice()
        if choices:
            ids = [c.id for c in choices]
            labels_by_id = {c.id: c.label for c in choices}
            model_choice_id = st.selectbox(
                "Model",
                options=ids,
                index=ids.index(default.id) if default and default.id in ids else 0,
                format_func=lambda i: labels_by_id.get(i, i),
                key="model_choice_id",
                help="Endpoints come from `.streamlit/secrets.toml` — see utils/model_registry.py to add one.",
            )
        else:
            model_choice_id = None

        with st.expander("Advanced"):
            detected_sentences = len(split_sentences(doc_text)) if doc_text else 0
            process_full_document = st.checkbox(
                "Process the full document", value=True, key="process_full_document",
                help="Turn this off to test only the first part and reduce LLM calls.",
            )
            if process_full_document:
                max_sentences = None
                st.caption(
                    f"All {detected_sentences} detected sentences will be processed "
                    f"({detected_sentences} LLM calls per run/model)."
                )
                if detected_sentences > 50:
                    st.warning("This is a large run. It may take several minutes and use many API calls.")
            else:
                max_sentences = st.number_input(
                    "Max sentences to process", min_value=1,
                    max_value=max(detected_sentences, 1),
                    value=min(25, max(detected_sentences, 1)), step=1,
                    key="sentence_limit",
                )
                st.caption(f"The document contains {detected_sentences} detected sentences.")
            temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1, key="temperature")
            st.caption(
                "How much the model is allowed to vary its answer. 0.0 means it always picks its "
                "most likely next word, so the same text gives the same annotations every time — "
                "which is what you want for a repeatable run. Higher values make it more "
                "adventurous and less consistent."
            )

    # -- Editable schema-derived prompt -------------------------------------
    st.divider()
    st.subheader(":material/description: Prompt", anchor=False)
    prompt_use_case = (
        USE_CASES.get("scientific_ie", selected_use_case)
        if current_source_meta.get("source_dataset") == "scirex"
        else selected_use_case
    )
    dataset_name = current_source_meta.get("source_dataset") or prompt_use_case.get("name")
    structured_method = METHODS[method_id]["structured"]
    suggested_prompt = suggest_prompt(
        st.session_state.labels,
        dataset_name,
        structured_method,
        label_definitions=prompt_use_case.get("label_definitions", {}),
        domain_guidance=prompt_use_case.get("domain_guidance", ""),
    )
    prompt_signature = json.dumps(
        [prompt_use_case.get("id"), dataset_name, method_id, sorted(st.session_state.labels)], sort_keys=True
    )
    if st.session_state.get("prompt_suggestion_signature") != prompt_signature:
        st.session_state.prompt_template_editor = suggested_prompt
        st.session_state.prompt_suggestion_signature = prompt_signature
    st.caption(
        "This suggestion is generated locally from the dataset and selected labels. "
        "Review or edit it before running; no LLM call is used to create it."
    )
    prompt_template = st.text_area(
        "Prompt template",
        key="prompt_template_editor",
        height=320,
        help="Keep the {sentence} and {indexed_tokens} placeholders so each document sentence can be inserted.",
    )
    st.button(
        ":material/refresh: Restore suggestion", key="restore_prompt_btn",
        on_click=_reset_prompt, args=(suggested_prompt,),
    )
    missing_placeholders = [
        placeholder for placeholder in ("{sentence}", "{indexed_tokens}")
        if placeholder not in prompt_template
    ]
    if missing_placeholders:
        st.error("Prompt must keep these placeholders: " + ", ".join(missing_placeholders))

    # -- Optional ticket routing --------------------------------------------
    st.divider()
    st.subheader(":material/account_tree: Optional routing", anchor=False)
    enable_routing = st.checkbox(
        "Suggest a destination department before extraction",
        key="enable_routing",
        help=(
            "Choose either an editable LLM classifier or a local nearest-neighbor classifier. "
            "The suggestion is final only after a reviewer approves or changes it."
        ),
    )
    routing_config = selected_use_case.get("routing", {})
    departments_text = st.text_area(
        "Allowed departments (one per line)",
        key="routing_departments_editor",
        height=150,
        disabled=not enable_routing,
    )
    departments = list(dict.fromkeys(
        line.strip() for line in departments_text.splitlines() if line.strip()
    ))
    routing_classifier = "llm"
    routing_prompt_template = ""
    routing_prompt_invalid = False
    embedding_ready = True
    embedding_top_k = 5
    if enable_routing:
        routing_classifier = st.radio(
            "Department classification method",
            options=["llm", "embedding"],
            format_func=lambda value: {
                "llm": "LLM classifier",
                "embedding": "Local embedding classifier",
            }[value],
            horizontal=True,
            key="routing_classifier",
            help=(
                "This choice affects department routing only. Entity extraction below still "
                "uses the selected LLM."
            ),
        )

        if routing_classifier == "llm":
            routing_suggestion = suggest_routing_prompt(
                departments,
                use_case_name=selected_use_case.get("name", "documents"),
                guidance=routing_config.get("guidance", ""),
            )
            routing_signature = json.dumps(
                [selected_use_case.get("id"), departments, routing_config.get("guidance", "")],
                sort_keys=True,
            )
            if st.session_state.get("routing_prompt_signature") != routing_signature:
                st.session_state.routing_prompt_editor = routing_suggestion
                st.session_state.routing_prompt_signature = routing_signature
            with st.expander("Review routing prompt", expanded=False):
                routing_prompt_template = st.text_area(
                    "Routing prompt template", key="routing_prompt_editor", height=280,
                    help=(
                        "Keep the {ticket_text} placeholder. The result must use one allowed "
                        "department."
                    ),
                )
            routing_prompt_invalid = "{ticket_text}" not in routing_prompt_template
            st.caption("LLM routing adds one model call per ticket and returns a quoted rationale.")
        else:
            embedding_status = embedding_routing_status(departments)
            embedding_ready = embedding_status.available
            if embedding_ready:
                st.success(embedding_status.message, icon=":material/check_circle:")
            else:
                st.info(embedding_status.message, icon=":material/info:")
            embedding_top_k = st.slider(
                "Reviewed neighbors used for routing",
                min_value=1,
                max_value=10,
                value=5,
                key="embedding_top_k",
                help=(
                    "The classifier embeds this ticket, retrieves similar reviewed tickets, "
                    "and applies a similarity-weighted department vote. Tune this on development data."
                ),
            )
            with st.expander("Build or update the reviewed-ticket index", expanded=not embedding_ready):
                st.caption(
                    "Upload a CSV containing reviewed ticket IDs, ticket text, and final department. "
                    "The CSV and derived index stay in the mounted local data directory and are not "
                    "sent to an LLM by this classifier."
                )
                if embedding_status.dependencies_available:
                    reference_upload = st.file_uploader(
                        "Reviewed reference CSV",
                        type=["csv"],
                        key="embedding_reference_upload",
                    )
                    if reference_upload is not None:
                        try:
                            reference_frame = pd.read_csv(io.BytesIO(reference_upload.getvalue()))
                        except Exception as exc:  # noqa: BLE001 - explain malformed uploads in the UI
                            st.error(f"Could not read the reference CSV: {exc}")
                            reference_frame = pd.DataFrame()
                        if not reference_frame.empty:
                            reference_columns = list(reference_frame.columns)
                            default_id = next(
                                (c for c in ("ticket_id", "id", "key") if c in reference_columns),
                                reference_columns[0],
                            )
                            default_text = next(
                                (c for c in ("text", "ticket_text", "description") if c in reference_columns),
                                reference_columns[0],
                            )
                            default_department = next(
                                (c for c in ("department", "gold_queue", "queue", "label") if c in reference_columns),
                                reference_columns[-1],
                            )
                            map_id, map_text, map_department = st.columns(3)
                            id_column = map_id.selectbox(
                                "ID column", reference_columns,
                                index=reference_columns.index(default_id), key="embedding_id_column",
                            )
                            text_column = map_text.selectbox(
                                "Text column", reference_columns,
                                index=reference_columns.index(default_text), key="embedding_text_column",
                            )
                            department_column = map_department.selectbox(
                                "Department column", reference_columns,
                                index=reference_columns.index(default_department),
                                key="embedding_department_column",
                            )
                            if st.button(
                                ":material/database: Build or replace index",
                                key="build_embedding_index_btn",
                            ):
                                try:
                                    with st.spinner("Encoding reviewed tickets..."):
                                        manifest = build_routing_index(
                                            reference_frame.to_dict("records"),
                                            id_column=id_column,
                                            text_column=text_column,
                                            department_column=department_column,
                                            allowed_departments=departments,
                                        )
                                except Exception as exc:  # noqa: BLE001 - keep setup recoverable
                                    st.error(f"Could not build the embedding index: {exc}")
                                else:
                                    st.success(
                                        f"Indexed {manifest['record_count']} reviewed tickets."
                                    )
                                    st.rerun()
                else:
                    st.code(
                        "ANNOTATION_TARGET=runtime-embeddings\n"
                        "docker-compose up --build annotation-tool",
                        language="bash",
                    )
            st.caption(
                "This baseline does not train a model and makes zero routing LLM calls. "
                "It builds a searchable index from human-reviewed examples."
            )

    if enable_routing and not departments:
        st.error("Add at least one allowed department or turn routing off.")
    if routing_prompt_invalid:
        st.error("Routing prompt must keep the {ticket_text} placeholder.")
    if enable_routing:
        st.caption("Routing is a reviewer-approved classification suggestion, not automatic ground truth.")

    # -- Uncertainty settings -------------------------------------------------
    # Its own row because it is the setting that decides how much the run
    # costs, and how much of the reviewer's time it will save afterwards.
    st.divider()
    col_est, col_est_opts = st.columns([2, 2])
    with col_est:
        estimator = st.selectbox(
            "Confidence scoring",
            options=list(ESTIMATORS.keys()),
            format_func=lambda e: ESTIMATORS[e]["label"],
            index=list(ESTIMATORS.keys()).index("logprob"),
            key="estimator",
            help="Scores how sure the model was about each entity, so you only have to check the doubtful ones.",
        )
        st.caption(ESTIMATORS[estimator]["description"])

    n_samples, sample_temperature, compare_ids = 3, 0.7, []
    with col_est_opts:
        if estimator == "self_consistency":
            n_samples = st.slider("Runs per sentence (K)", 2, 7, 3, key="n_samples")
            sample_temperature = st.slider("Sampling temperature", 0.1, 1.0, 0.7, 0.1, key="sample_temp")
            st.caption(f"Costs about {n_samples}x a normal run.")
        elif estimator == "model_agreement":
            ids = [c.id for c in choices]
            labels_by_id = {c.id: c.label for c in choices}
            compare_ids = st.multiselect(
                "Models to compare (pick 2 or more)",
                options=ids,
                default=ids[:2],
                format_func=lambda i: labels_by_id.get(i, i),
                key="compare_ids",
            )
            st.caption(f"Costs about {max(len(compare_ids), 1)}x a normal run.")
        elif estimator == "logprob":
            st.caption(
                "No extra cost. If the endpoint does not return log-probabilities, the run still "
                "works and the confidence column stays empty."
            )

    # -- How do we turn those scores into a review queue? ---------------------
    # Two ways of deciding which rows a human sees. "Budget" ranks the entities
    # and shows the least confident slice; "threshold" flags everything below a
    # fixed number. Budget is the default because the estimators put their
    # scores on very different scales, so one fixed number cannot suit both.
    if estimator == "none":
        # No scoring means no scores to rank, so a "review the least confident
        # 20%" slider would be asking a question this run cannot answer.
        # We hide the controls and explain why instead of showing a dead widget.
        st.info(
            "Confidence scoring is off, so every entity will be listed for review. "
            "Pick an estimator above if you want the doubtful ones separated out.",
            icon=":material/info:",
        )
        flag_mode = "budget"
        review_budget = 1.0          # 1.0 = everything, i.e. no filtering
        review_threshold = DEFAULT_REVIEW_THRESHOLD
    else:
        col_mode, col_amount = st.columns([2, 2])

        with col_mode:
            flag_mode = st.radio(
                "How to pick rows for review",
                options=list(FLAG_MODES.keys()),
                format_func=lambda m: FLAG_MODES[m]["label"],
                horizontal=True,
                key="flag_mode",
            )
            st.caption(FLAG_MODES[flag_mode]["description"])

        with col_amount:
            if flag_mode == "budget":
                # Percent as a whole number (5, 10, 15...). The first version
                # used floats 0.0-1.0 with a "%.0f%%" format, which printed the
                # raw float, so the slider read "0%" on the left and "1%" on
                # the right. Integers make the widget say what it means.
                review_budget_pct = st.slider(
                    "Share of entities to review",
                    min_value=5, max_value=100,
                    value=int(DEFAULT_REVIEW_BUDGET * 100), step=5,
                    format="%d%%",
                    key="review_budget_pct",
                    help=(
                        "We sort the entities from least to most confident and put this share in "
                        "front of you. 20% means you read the doubtful fifth and trust the rest."
                    ),
                )
                review_budget = review_budget_pct / 100
                review_threshold = DEFAULT_REVIEW_THRESHOLD
            else:
                review_threshold = st.slider(
                    "Flag everything below this confidence",
                    min_value=0.0, max_value=1.0,
                    value=DEFAULT_REVIEW_THRESHOLD, step=0.05,
                    key="review_threshold",
                    help=(
                        f"Our default is {HIGH_UNCERTAINTY_BELOW:.2f}. Careful with token "
                        "confidence: those scores sit very close to 1.0, so a cut-off like 0.80 "
                        "can flag nothing at all."
                    ),
                )
                review_budget = DEFAULT_REVIEW_BUDGET

    run_disabled = not doc_text or not st.session_state.labels or not is_configured()
    if missing_placeholders:
        run_disabled = True
    if enable_routing and (
        not departments
        or (routing_classifier == "llm" and routing_prompt_invalid)
        or (routing_classifier == "embedding" and not embedding_ready)
    ):
        run_disabled = True
    if estimator == "model_agreement" and len(compare_ids) < 2:
        run_disabled = True
        st.info("Pick at least two models to compare.", icon=":material/info:")
    if not is_configured():
        st.warning(missing_credentials_message(), icon=":material/warning:")
    run_clicked = st.button(
        ":material/auto_awesome: Run LLM pre-annotation", type="primary", disabled=run_disabled, width="stretch",
        key="run_button",
    )

if run_clicked:
    status_box = st.status("Running workflow...", expanded=True)

    def _progress(i, n, sentence_text):
        status_box.write(f"Sentence {i + 1}/{n}: _{sentence_text[:70]}{'...' if len(sentence_text) > 70 else ''}_")

    base_choice = choice_by_id(model_choice_id)
    routing_result = None
    routing_error = None
    if enable_routing:
        status_box.write(
            "Classifying the destination with reviewed embedding neighbors..."
            if routing_classifier == "embedding"
            else "Suggesting a destination department with the LLM..."
        )
        try:
            if routing_classifier == "embedding":
                routing_result = route_ticket_with_embeddings(
                    ticket_text=doc_text,
                    departments=departments,
                    top_k=embedding_top_k,
                    ticket_id=current_source_meta.get("ticket_id"),
                )
            else:
                routing_result = route_ticket(
                    ticket_text=doc_text,
                    departments=departments,
                    prompt_template=routing_prompt_template,
                    choice=base_choice,
                    llm_params={
                        "temperature": 0.0,
                        "max_tokens": 300,
                        "timeout": 45,
                        "max_retries": 1,
                    },
                )
            routing_result["allowed_departments"] = departments
            status_box.write(f"Routing suggestion: **{routing_result['department']}**")
        except Exception as exc:  # noqa: BLE001 - optional routing must not prevent extraction
            routing_error = f"{type(exc).__name__}: {exc}"
            status_box.write(f"Routing failed ({routing_error}). Continuing with extraction.")
    try:
        passes, want_logprobs = build_passes(
            estimator=estimator,
            base_choice=base_choice,
            n_samples=n_samples,
            sample_temperature=sample_temperature,
            compare_choices=[choice_by_id(i) for i in compare_ids],
        )
        result = run_annotation(
            text=doc_text,
            labels=st.session_state.labels,
            method_id=method_id,
            max_sentences=max_sentences,
            llm_params={"temperature": temperature, "max_tokens": 500, "timeout": 45, "max_retries": 1},
            progress_callback=_progress,
            estimator=estimator,
            passes=passes,
            want_logprobs=want_logprobs,
            prompt_template=prompt_template,
        )
    except Exception as exc:  # noqa: BLE001 - show any LLM/config error to the user instead of crashing
        status_box.update(label="Failed", state="error")
        st.error(f"Extraction failed: {exc}")
        st.stop()

    routing_calls = int(enable_routing and routing_classifier == "llm")
    status_box.update(
        label=(
            f"Done — {len(result['entities'])} entities from "
            f"{result['n_llm_calls'] + routing_calls} total LLM calls"
        ),
        state="complete",
    )
    if estimator == "logprob" and not result["logprobs_available"]:
        st.warning(
            "This endpoint did not return log-probabilities, so there are no confidence scores. "
            "Use **Self-consistency** or **Model agreement** instead — both work on any endpoint.",
            icon=":material/info:",
        )

    entities = [
        {
            "id": new_entity_id(), "text": e["text"], "type": e["type"], "start": e["start"], "end": e["end"],
            "source": "model", "model_name": base_choice.id if base_choice else None, "status": "pending",
            "confidence": e.get("confidence"), "conf_source": e.get("conf_source"), "voters": e.get("voters"),
            "original_text": e["text"], "original_type": e["type"],
            "original_start": e["start"], "original_end": e["end"],
        }
        for e in result["entities"]
    ]
    st.session_state.entities = entities
    st.session_state.doc_text = doc_text
    st.session_state.doc_name = doc_name
    st.session_state.routing_result = (
        routing_result if routing_result is not None
        else ({"error": routing_error, "allowed_departments": departments} if enable_routing else None)
    )
    st.session_state.routing_approved_department = (
        routing_result.get("department") if routing_result else None
    )
    routing_input_tokens = routing_result.get("input_tokens", 0) if routing_result else 0
    routing_output_tokens = routing_result.get("output_tokens", 0) if routing_result else 0
    routing_total_tokens = routing_result.get("total_tokens", 0) if routing_result else 0
    st.session_state.last_run_meta = {
        "method": method_id,
        "n_sentences": result["n_sentences"],
        "n_sentences_total": result["n_sentences_total"],
        "estimator": estimator,
        "n_passes": result["n_passes"],
        "n_llm_calls": result["n_llm_calls"],
        "routing_llm_calls": routing_calls,
        "routing_classifier": routing_classifier if enable_routing else None,
        "total_llm_calls": result["n_llm_calls"] + routing_calls,
        "pass_ids": result["pass_ids"],
        "logprobs_available": result["logprobs_available"],
        "review_threshold": review_threshold,
        "flag_mode": flag_mode,
        "review_budget": review_budget,
        "processed_char_end": result.get("processed_char_end", 0),
        "prompt_sha256": prompt_hash(prompt_template),
        "input_tokens": result.get("input_tokens", 0) + routing_input_tokens,
        "output_tokens": result.get("output_tokens", 0) + routing_output_tokens,
        "total_tokens": result.get("total_tokens", 0) + routing_total_tokens,
        "usage_reported": (
            result.get("usage_reported", False)
            and (
                not enable_routing
                or bool(routing_result and routing_result.get("usage_reported", False))
            )
        ),
        "routing_prompt_sha256": (
            prompt_hash(routing_prompt_template)
            if enable_routing and routing_classifier == "llm"
            else None
        ),
        "routing_retrieval": routing_result.get("retrieval") if routing_result else None,
        "routing_error": routing_error,
        "use_case_id": use_case_id,
        "source": current_source_meta,
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

    routing_review = {}
    saved_routing = st.session_state.get("routing_result")
    if saved_routing:
        with st.container(border=True):
            st.markdown("#### :material/account_tree: Review destination")
            if saved_routing.get("error"):
                st.warning(
                    "No routing suggestion was produced, but entity extraction succeeded. "
                    f"Details: {saved_routing['error']}"
                )
                routing_review = {
                    "task": "department_routing",
                    "review_status": "routing_failed",
                    "error": saved_routing["error"],
                }
            else:
                predicted_department = saved_routing["department"]
                allowed_departments = saved_routing.get("allowed_departments", [predicted_department])
                if st.session_state.routing_approved_department not in allowed_departments:
                    st.session_state.routing_approved_department = predicted_department
                route_col1, route_col2 = st.columns([1, 2])
                classifier_name = saved_routing.get("classifier", "llm")
                route_col1.metric("Classifier suggestion", predicted_department)
                approved_department = route_col2.selectbox(
                    "Reviewer-approved department",
                    options=allowed_departments,
                    key="routing_approved_department",
                )
                if saved_routing.get("evidence"):
                    st.caption(f"Evidence: “{saved_routing['evidence']}”")
                if saved_routing.get("reason"):
                    st.caption(f"Reason: {saved_routing['reason']}")
                if classifier_name == "embedding_nearest_neighbor":
                    confidence = saved_routing.get("confidence")
                    if confidence is not None:
                        st.caption(f"Embedding vote share: {confidence:.1%}")
                    hits = saved_routing.get("retrieval", {}).get("hits", [])
                    if hits:
                        with st.expander("Retrieved reviewed tickets", expanded=False):
                            st.dataframe(pd.DataFrame(hits), hide_index=True, width="stretch")
                routing_review = {
                    "task": "department_routing",
                    "model_prediction": predicted_department,
                    "approved_department": approved_department,
                    "review_status": (
                        "confirmed" if approved_department == predicted_department else "corrected"
                    ),
                    "reason": saved_routing.get("reason", ""),
                    "evidence": saved_routing.get("evidence", ""),
                    "evidence_valid": saved_routing.get("evidence_valid", False),
                    "classifier": classifier_name,
                    "confidence": saved_routing.get("confidence"),
                    "retrieval": saved_routing.get("retrieval"),
                    "model": saved_routing.get("model"),
                }

    numbers = assign_span_numbers(entities)
    visible = [e for e in entities if e["status"] != "deleted"]
    pending = sum(1 for e in visible if e["status"] == "pending")
    run_meta = st.session_state.last_run_meta or {}
    threshold = run_meta.get("review_threshold", DEFAULT_REVIEW_THRESHOLD)
    mode = run_meta.get("flag_mode", "budget")
    budget = run_meta.get("review_budget", DEFAULT_REVIEW_BUDGET)
    stats = summarise(visible, threshold=threshold, mode=mode, budget=budget)
    flagged_ids = {visible[i]["id"] for i in stats["flagged_idx"]}

    # The document with the entities highlighted. We use st.markdown rather
    # than st.components.v1.html on purpose: components.html renders inside an
    # iframe, and a link in the table below cannot jump to an anchor that lives
    # in a different document. Plain markdown keeps the anchors on this page.
    focus_id = st.session_state.get("focus_span_id")
    st.markdown(
        render_highlighted_html(
            st.session_state.doc_text, entities, labels,
            numbers=numbers, focus_id=focus_id, flagged_ids=flagged_ids,
        ),
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Entities", len(visible))
    m2.metric("Pending review", pending)
    m3.metric("Confirmed", sum(1 for e in visible if e["status"] == "confirmed"))
    m4.metric("Relabeled", sum(1 for e in visible if e["status"] == "edited"))
    m5.metric(
        "Needs judgement",
        stats["n_flagged"],
        help=(
            f"The least confident {budget:.0%} of entities, plus any without a score."
            if mode == "budget"
            else f"Entities scored below {threshold:.2f}, or with no score at all."
        ),
    )

    if stats["n_scored"] and not has_signal(visible):
        st.info(
            "The model gave every entity the same confidence, so there is nothing to rank here. "
            "That usually means the text was easy. To get a signal on harder text, try "
            "**Self-consistency** with a higher temperature, or **Model agreement**.",
            icon=":material/info:",
        )
    elif stats["n_scored"]:
        saved = len(visible) - stats["n_flagged"]
        st.success(
            f"**{stats['n_flagged']} of {len(visible)} entities** are worth your attention. "
            f"The other {saved} are in the collapsed section below if you want to check them too.",
            icon=":material/filter_alt:",
        )

    st.caption(
        "Tick **confirmed** once you've checked a row, change **label** to relabel it, or tick "
        "**delete** to remove it, then click **Save corrections**. Click the **#** of any row to "
        "jump to that word in the text above."
    )

    # ---------------------------------------------------------------
    # Split the table in two: what needs attention, and the rest.
    # ---------------------------------------------------------------
    full_df = entities_to_dataframe(entities, numbers=numbers)
    flagged_df = full_df[full_df["id"].isin(flagged_ids)].reset_index(drop=True)
    rest_df = full_df[~full_df["id"].isin(flagged_ids)].reset_index(drop=True)

    column_config = {
        "id": None,   # internal, used to match rows back up
        "num": None,  # the number is shown by the "jump" link column instead
        "jump": st.column_config.LinkColumn(
            "#", width="small", display_text=r"#span-(.*)",
            help="Click to jump to this entity in the text above.",
        ),
        "text": st.column_config.TextColumn("Text", disabled=True),
        "type": st.column_config.SelectboxColumn("Label", options=labels),
        "confidence": st.column_config.ProgressColumn(
            "Confidence", min_value=0.0, max_value=1.0, format="%.2f",
            help=(
                "How sure the model was. Empty means it could not be scored. This is a ranking "
                "signal, not a calibrated probability: read a 0.80 as 'more reliable than a 0.40', "
                "not as '80% likely to be right'."
            ),
        ),
        "start": st.column_config.NumberColumn("Start", disabled=True),
        "end": st.column_config.NumberColumn("End", disabled=True),
        "source": st.column_config.TextColumn(
            "From", disabled=True,
            help="Which model suggested this entity, or 'human' if you added it yourself.",
        ),
        "status": st.column_config.TextColumn("Status", disabled=True),
        "confirmed": st.column_config.CheckboxColumn("Confirmed"),
        "delete": st.column_config.CheckboxColumn("Delete"),
    }

    version = st.session_state.entities_version
    st.markdown("##### :material/psychology: Where we need your judgement")
    if flagged_df.empty:
        st.caption(
            "Nothing stood out as doubtful, so the model is equally sure about everything it "
            "found. The full list is in the section below."
        )
        edited_flagged = flagged_df
    else:
        edited_flagged = st.data_editor(
            flagged_df, column_config=column_config, hide_index=True,
            width="stretch", key=f"entity_editor_flagged_{version}",
        )

    with st.expander(
        f":material/inventory_2: The remaining {len(rest_df)} entities "
        "the model was confident about — open if you want to check them too",
        expanded=flagged_df.empty,
    ):
        if rest_df.empty:
            st.caption("Nothing here.")
            edited_rest = rest_df
        else:
            edited_rest = st.data_editor(
                rest_df, column_config=column_config, hide_index=True,
                width="stretch", key=f"entity_editor_rest_{version}",
            )

    # Both tables edit the same entity list, so the save button below needs
    # them stitched back together.
    import pandas as _pd
    edited_df = _pd.concat([edited_flagged, edited_rest], ignore_index=True)

    col_save, col_conf_high, col_confirm, col_reset = st.columns(4)
    if col_save.button(":material/save: Save corrections", width="stretch", key="save_corrections_btn"):
        st.session_state.entities = apply_edits(entities, edited_df, labels)
        bump_entities_version()
        st.rerun()

    # The button that turns the confidence score into saved time: accept
    # everything the model was sure about and leave only the doubtful rows.
    if col_conf_high.button(
        ":material/bolt: Accept confident rows", width="stretch", key="confirm_confident_btn",
        help="Confirms every pending entity that was not flagged, leaving the flagged ones for you.",
        disabled=not stats["n_scored"],
    ):
        n_confirmed = 0
        for e in st.session_state.entities:
            if e["status"] == "pending" and e["id"] not in flagged_ids:
                e["status"] = "confirmed"
                n_confirmed += 1
        bump_entities_version()
        st.toast(f"Confirmed {n_confirmed} confident entities", icon=":material/bolt:")
        st.rerun()

    if col_confirm.button(":material/done_all: Confirm all pending", width="stretch", key="confirm_all_btn"):
        n_before = sum(1 for e in entities if e["status"] == "pending")
        st.session_state.entities = confirm_all_pending(entities)
        bump_entities_version()
        st.toast(f"Confirmed {n_before} pending entities", icon=":material/done_all:")
        st.rerun()
    col_reset.button(
        ":material/refresh: Clear annotation",
        width="stretch",
        key="clear_btn",
        on_click=_clear_annotation,
    )

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
        run_meta={
            key: value for key, value in (st.session_state.last_run_meta or {}).items()
            if key != "source"
        },
        source_meta=(st.session_state.last_run_meta or {}).get("source", {}),
        classification=routing_review,
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
