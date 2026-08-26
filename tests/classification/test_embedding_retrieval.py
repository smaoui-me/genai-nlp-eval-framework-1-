"""Network-free tests for the optional classification retrieval index."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genai_eval.classification.retrieval.embedding_index import EmbeddingIndex, write_index  # noqa: E402


def make_index(tmp_path: Path) -> EmbeddingIndex:
    source = tmp_path / "reference.csv"
    source.write_text("ticket_id,text\n1,VPN failure\n2,Invoice error\n3,Password reset\n", encoding="utf-8")
    records = [
        {"ticket_id": "1", "text": "VPN failure", "type": "Incident", "queue": "IT", "tags": ["VPN"]},
        {"ticket_id": "2", "text": "Invoice error", "type": "Problem", "queue": "Billing", "tags": ["Invoice"]},
        {"ticket_id": "3", "text": "Password reset", "type": "Request", "queue": "IT", "tags": ["Access"]},
    ]
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32)
    write_index(
        tmp_path / "index",
        records,
        embeddings,
        "fake-encoder",
        source,
        {"ticket_id": "ticket_id", "text": "text"},
    )
    return EmbeddingIndex(tmp_path / "index")


def test_embedding_index_retrieves_nearest_and_excludes_current_record(tmp_path):
    index = make_index(tmp_path)
    hits = index.query_by_vector(
        np.asarray([1.0, 0.0]),
        top_k=2,
        exclude_ticket_id="1",
        exclude_text="VPN failure",
    )
    assert [hit.ticket_id for hit in hits] == ["3", "2"]
    assert hits[0].similarity > hits[1].similarity
    assert index.encoder_model == "fake-encoder"


def test_embedding_index_rejects_tampered_records(tmp_path):
    index = make_index(tmp_path)
    records_path = index.index_dir / "records.jsonl"
    with records_path.open("a", encoding="utf-8") as target:
        target.write(json.dumps({"ticket_id": "tampered"}) + "\n")
    with pytest.raises(ValueError, match="checksum"):
        EmbeddingIndex(index.index_dir)


def test_embedding_index_rejects_wrong_query_dimension(tmp_path):
    index = make_index(tmp_path)
    with pytest.raises(ValueError, match="dimension"):
        index.query_by_vector(np.asarray([1.0, 0.0, 0.0]), top_k=1)
