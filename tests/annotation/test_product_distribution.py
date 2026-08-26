import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "genai-nlp-annotation-tool"
sys.path.insert(0, str(APP))

from utils.annotation_store import build_gold_export  # noqa: E402
from utils.prompt_builder import suggest_prompt  # noqa: E402
from utils import ticket_routing  # noqa: E402
from utils.use_case_config import load_use_cases  # noqa: E402


def test_ticket_template_builds_domain_prompt():
    templates = load_use_cases()
    ticket = templates["ticket_support"]
    assert ticket["routing"]["enabled_by_default"] is True
    assert "Technical Support" in ticket["routing"]["departments"]
    prompt = suggest_prompt(
        ticket["entity_labels"], ticket["name"], True,
        label_definitions=ticket["label_definitions"],
        domain_guidance=ticket["domain_guidance"],
    )
    assert "ErrorCode: An exact error identifier" in prompt
    assert "Preserve exact identifiers" in prompt
    assert "{sentence}" in prompt and "{indexed_tokens}" in prompt


def test_ticket_routing_is_allowed_evidence_carrying_and_metered(monkeypatch):
    response = SimpleNamespace(
        text='```json\n{"department":"technical support","reason":"VPN access",'
             '"evidence":"cannot connect to the corporate VPN"}\n```',
        model_id="fake:model", input_tokens=40, output_tokens=12,
        total_tokens=52, usage_reported=True,
    )
    monkeypatch.setattr(ticket_routing, "call_llm_full", lambda *args, **kwargs: response)
    text = "Maya cannot connect to the corporate VPN after a password reset."
    result = ticket_routing.route_ticket(
        text,
        ["IT Support", "Technical Support"],
        ticket_routing.suggest_routing_prompt(["IT Support", "Technical Support"]),
        choice=SimpleNamespace(id="fake:model"),
    )
    assert result["department"] == "Technical Support"
    assert result["classifier"] == "llm"
    assert result["evidence_valid"] is True
    assert result["total_tokens"] == 52


def test_ticket_routing_rejects_unconfigured_department():
    with pytest.raises(ValueError, match="outside the allowed list"):
        ticket_routing.validate_routing(
            {"department": "Legal", "reason": "x", "evidence": "invoice"},
            ["Billing"], "invoice problem",
        )


def test_export_keeps_reviewed_classification_separate_from_entities():
    classification = {
        "task": "department_routing", "model_prediction": "IT Support",
        "approved_department": "Technical Support", "review_status": "corrected",
    }
    export = build_gold_export(
        "ticket-1", "VPN unavailable", ["Issue"], "few_shot_structured", [],
        classification=classification,
    )
    assert export["classification"] == classification
    assert export["gold_entities"] == []


def test_product_container_and_synthetic_data_are_present():
    config = json.loads((ROOT / ".devcontainer" / "devcontainer.json").read_text(encoding="utf-8"))
    assert 8501 in config["forwardPorts"]
    assert "requirements-dev.txt" in config["postCreateCommand"]

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim-bookworm" in dockerfile
    assert "annotation-tool:" in compose and "${APP_PORT:-8501}:8501" in compose
    assert '${ANNOTATION_TARGET:-runtime-embeddings}' in compose
    assert all(
        target in compose
        for target in ("target: evaluation", "target: test", "target: embeddings")
    )
    assert "FROM embeddings AS runtime-embeddings" in dockerfile
    assert "**/.streamlit/secrets.toml" in dockerignore

    eval_requirements = (ROOT / "requirements-eval.txt").read_text(encoding="utf-8")
    dev_requirements = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "scikit-learn==" in eval_requirements
    assert "-r requirements-eval.txt" in dev_requirements

    embedding_requirements = (ROOT / "requirements-embeddings.txt").read_text(encoding="utf-8")
    assert "sentence-transformers==" in embedding_requirements

    sample = (ROOT / "sample_data" / "tickets.csv").read_text(encoding="utf-8")
    assert "TCK-1001" in sample and "ticket_id,title,text" in sample
