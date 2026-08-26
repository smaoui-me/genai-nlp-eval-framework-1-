import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "genai-nlp-annotation-tool"))

from utils import embedding_routing


def test_similarity_weighted_vote_uses_reviewed_department_labels():
    hits = [
        SimpleNamespace(queue_label="IT Support", similarity=0.80),
        SimpleNamespace(queue_label="Billing", similarity=0.70),
        SimpleNamespace(queue_label="Billing", similarity=0.65),
        SimpleNamespace(queue_label="Unknown", similarity=0.99),
    ]

    department, confidence = embedding_routing._weighted_department_vote(
        hits, ["IT Support", "Billing"]
    )

    assert department == "Billing"
    assert confidence == pytest.approx(1.35 / 2.15)


def test_embedding_status_explains_missing_optional_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(embedding_routing, "_embedding_dependencies_available", lambda: False)

    status = embedding_routing.embedding_routing_status(["IT Support"], index_dir=tmp_path)

    assert status.available is False
    assert status.dependencies_available is False
    assert "runtime-embeddings" in status.message


def test_weighted_vote_fails_when_index_labels_do_not_match_active_schema():
    hits = [SimpleNamespace(queue_label="Legal", similarity=0.9)]

    with pytest.raises(ValueError, match="active allowed departments"):
        embedding_routing._weighted_department_vote(hits, ["IT Support"])
