"""Render every page with Streamlit's AppTest to catch UI-level breakage.

No browser and no LLM calls: nothing clicks 'Run', so the pages only render.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

from streamlit.testing.v1 import AppTest

FAILS = []
def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond: FAILS.append(name)

PAGES = [
    ("app.py", "router"),
    ("pages/0_home.py", "Home"),
    ("pages/1_annotate.py", "Annotate"),
    ("pages/2_annotation_evaluation.py", "Annotation eval"),
    ("pages/3_model_comparison.py", "Model comparison"),
]

for path, name in PAGES:
    at = AppTest.from_file(str(ROOT / path), default_timeout=60)
    try:
        at.run()
    except Exception as exc:  # noqa: BLE001
        check(f"{name} renders", False, f"raised {type(exc).__name__}: {exc}")
        continue
    if at.exception:
        msgs = [str(e.value)[:200] for e in at.exception]
        check(f"{name} renders", False, f"exception: {msgs}")
    else:
        check(f"{name} renders", True)

# --- Annotate page specifics ------------------------------------------------
print("\n=== Annotate page widgets ===")
at = AppTest.from_file(str(ROOT / "pages/1_annotate.py"), default_timeout=60)
at.run()
if not at.exception:
    sel_keys = {s.key for s in at.selectbox}
    use_case = next(s for s in at.selectbox if s.key == "use_case_id")
    check("ticket product template is the default", use_case.value == "ticket_support", str(use_case.value))
    ticket_labels = next(m for m in at.multiselect if m.key == "labels_select")
    check("ticket extraction schema loads", {"Product", "Issue", "ErrorCode", "RequestedAction"}.issubset(ticket_labels.value), str(ticket_labels.value))
    routing_toggle = next(c for c in at.checkbox if c.key == "enable_routing")
    check("ticket routing is offered before extraction", routing_toggle.value is True)
    routing_classifier = next(r for r in at.radio if r.key == "routing_classifier")
    check("LLM and embedding routing are selectable",
          set(routing_classifier.options) == {"LLM classifier", "Local embedding classifier"},
          str(routing_classifier.options))
    routing_classifier.set_value("embedding").run()
    check("embedding neighbor count is configurable", "embedding_top_k" in {s.key for s in at.slider})
    check("embedding routing setup renders without exception", not at.exception, str(at.exception))
    next(r for r in at.radio if r.key == "routing_classifier").set_value("llm").run()
    check("routing departments are configurable", "routing_departments_editor" in {t.key for t in at.text_area})
    check("model dropdown present", "model_choice_id" in sel_keys, str(sorted(k for k in sel_keys if k)))
    check("confidence estimator dropdown present", "estimator" in sel_keys)
    slider_keys = {s.key for s in at.slider}
    # Default is budget mode, so the amount slider is review_budget_pct
    # (whole percent, 5-100 — floats made the widget read "0%" to "1%").
    check("review budget slider present", "review_budget_pct" in slider_keys, str(sorted(k for k in slider_keys if k)))
    full_document = next(c for c in at.checkbox if c.key == "process_full_document")
    check("full-document processing defaults on", full_document.value is True)
    pct = [s for s in at.slider if s.key == "review_budget_pct"][0]
    check("budget slider is a real percent scale", pct.min == 5 and pct.max == 100, f"{pct.min}-{pct.max}")
    check("budget slider defaults to 20%", pct.value == 20, str(pct.value))
    radio_keys = {r.key for r in at.radio}
    check("flag mode radio present", "flag_mode" in radio_keys, str(sorted(k for k in radio_keys if k)))

    # Switching to the fixed cut-off should swap in the threshold slider.
    [r for r in at.radio if r.key == "flag_mode"][0].set_value("threshold").run()
    check("threshold slider appears in threshold mode", "review_threshold" in {s.key for s in at.slider})
    check("no exception switching flag mode", not at.exception, str(at.exception))
    [r for r in at.radio if r.key == "flag_mode"][0].set_value("budget").run()

    # With scoring off there is nothing to rank, so the flag controls hide.
    [s for s in at.selectbox if s.key == "estimator"][0].set_value("none").run()
    check("no flag-mode radio when scoring is off", "flag_mode" not in {r.key for r in at.radio})
    check("explains why instead of showing a dead widget",
          any("scoring is off" in str(i.value) for i in at.info))
    check("no exception with scoring off", not at.exception, str(at.exception))
    [s for s in at.selectbox if s.key == "estimator"][0].set_value("logprob").run()

    est = [s for s in at.selectbox if s.key == "estimator"][0]
    check("estimator defaults to logprob (cheapest)", est.value == "logprob", str(est.value))
    check("all four estimators offered", len(est.options) == 4, str(est.options))

    source = next(r for r in at.radio if r.key == "source_mode")
    if "SciREX fixture" in source.options:
        source.set_value("SciREX fixture").run()
        check("SciREX example selector present", "scirex_example_id" in {s.key for s in at.selectbox})
        scirex_selector = next(s for s in at.selectbox if s.key == "scirex_example_id")
        check("prepared SciREX examples are available", len(scirex_selector.options) > 0,
              str(len(scirex_selector.options)))
        scirex_labels = next(m for m in at.multiselect if m.key == "labels_select")
        check("clear SciREX labels selected", set(scirex_labels.value) == {"Method", "Task", "Metric", "Dataset"}, str(scirex_labels.value))
        check("editable suggested prompt visible", "prompt_template_editor" in {t.key for t in at.text_area})
        check("SciREX source renders without exception", not at.exception, str(at.exception))
    else:
        print("  SKIP  SciREX UI checks (optional generated dataset is not in this image)")

    custom_label = "CustomTestLabel"
    next(t for t in at.text_input if t.key == "new_label_input").set_value(custom_label).run()
    next(b for b in at.button if b.key == "add_label_btn").click().run()
    labels = next(m for m in at.multiselect if m.key == "labels_select")
    check("custom label becomes available", custom_label in labels.options, str(labels.options))
    check("custom label is selected", custom_label in labels.value, str(labels.value))
    check("custom label addition has no exception", not at.exception, str(at.exception))

    # Switching to self-consistency should reveal the K slider.
    [s for s in at.selectbox if s.key == "estimator"][0].set_value("self_consistency").run()
    check("K slider appears for self-consistency", "n_samples" in {s.key for s in at.slider})
    check("no exception after switching", not at.exception, str(at.exception))

    # Switching to model agreement should reveal the model multiselect.
    [s for s in at.selectbox if s.key == "estimator"][0].set_value("model_agreement").run()
    check("model multiselect appears", "compare_ids" in {m.key for m in at.multiselect})
    check("no exception on model_agreement", not at.exception, str(at.exception))

# A completed optional routing call must render as an editable human decision,
# even when there are no extracted entities.
route_at = AppTest.from_file(str(ROOT / "pages/1_annotate.py"), default_timeout=60)
route_at.session_state["doc_text"] = "The VPN displays AUTH-403."
route_at.session_state["doc_name"] = "ticket-1"
route_at.session_state["entities"] = []
route_at.session_state["routing_result"] = {
    "department": "Technical Support", "reason": "VPN failure", "evidence": "AUTH-403",
    "evidence_valid": True, "model": "fake:model",
    "allowed_departments": ["IT Support", "Technical Support"],
}
route_at.session_state["routing_approved_department"] = "Technical Support"
route_at.session_state["last_run_meta"] = {"n_sentences": 1, "n_sentences_total": 1, "source": {}}
route_at.run()
check("reviewer can approve or correct ticket routing",
      "routing_approved_department" in {s.key for s in route_at.selectbox})
check("routing review renders without exception", not route_at.exception, str(route_at.exception))
next(b for b in route_at.button if b.key == "clear_btn").click().run()
check("clear annotation removes document state", route_at.session_state["doc_text"] == "")
check("clear annotation removes routing state", route_at.session_state["routing_result"] is None)
check("clear annotation does not mutate a rendered widget", not route_at.exception, str(route_at.exception))

print("\n=== Model comparison page ===")
at = AppTest.from_file(str(ROOT / "pages/3_model_comparison.py"), default_timeout=60)
at.run()
if not at.exception:
    # With one configured provider the page should explain itself, not crash.
    infos = [i.value for i in at.info] + [w.value for w in at.warning]
    has_guidance = any("Ollama" in str(t) or "compare" in str(t).lower() for t in infos)
    check("page gives guidance or renders setup", has_guidance or len(at.multiselect) > 0, str(infos)[:160])

print("\n" + "=" * 60)
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}"); sys.exit(1)
print("All app render checks passed.")
