"""Local nearest-neighbor ticket routing over reviewed reference examples."""

from __future__ import annotations

import csv
import importlib.util
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable


APP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_DIR.parent
DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "classification" / "retrieval" / "index"
DEFAULT_REFERENCE_PATH = (
    REPO_ROOT / "data" / "classification" / "retrieval" / "reference_tickets.csv"
)
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class EmbeddingRoutingStatus:
    available: bool
    dependencies_available: bool
    index_available: bool
    message: str
    encoder_model: str | None = None
    record_count: int = 0
    matching_departments: tuple[str, ...] = ()


def _embedding_dependencies_available() -> bool:
    try:
        return (
            importlib.util.find_spec("sentence_transformers") is not None
            and importlib.util.find_spec(
                "genai_eval.classification.retrieval.embedding_index"
            ) is not None
        )
    except ModuleNotFoundError:
        return False


def embedding_routing_status(
    departments: Iterable[str],
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> EmbeddingRoutingStatus:
    dependencies = _embedding_dependencies_available()
    manifest_path = index_dir / "manifest.json"
    if not dependencies:
        return EmbeddingRoutingStatus(
            available=False,
            dependencies_available=False,
            index_available=manifest_path.exists(),
            message=(
                "The local embedding runtime is not installed. In Docker, set "
                "ANNOTATION_TARGET=runtime-embeddings and rebuild the annotation-tool service."
            ),
        )
    if not manifest_path.exists():
        return EmbeddingRoutingStatus(
            available=False,
            dependencies_available=True,
            index_available=False,
            message="No reviewed ticket index exists yet. Upload a reference CSV below and build it.",
        )

    try:
        from genai_eval.classification.retrieval.embedding_index import EmbeddingIndex

        index = EmbeddingIndex(index_dir)
    except Exception as exc:  # noqa: BLE001 - return a useful UI state
        return EmbeddingRoutingStatus(
            available=False,
            dependencies_available=True,
            index_available=True,
            message=f"The embedding index is invalid: {type(exc).__name__}: {exc}",
        )

    by_normalized = {str(value).casefold(): str(value) for value in departments}
    indexed_queues = {str(record.get("queue", "")).casefold() for record in index.records}
    matching = tuple(
        department
        for normalized, department in by_normalized.items()
        if normalized in indexed_queues
    )
    if not matching:
        return EmbeddingRoutingStatus(
            available=False,
            dependencies_available=True,
            index_available=True,
            message=(
                "The index is valid, but none of its department labels match the active allowed "
                "departments. Rebuild it with the current routing schema."
            ),
            encoder_model=index.encoder_model,
            record_count=len(index.records),
        )
    return EmbeddingRoutingStatus(
        available=True,
        dependencies_available=True,
        index_available=True,
        message=(
            f"Ready: {len(index.records)} reviewed tickets; "
            f"{len(matching)} matching departments."
        ),
        encoder_model=index.encoder_model,
        record_count=len(index.records),
        matching_departments=matching,
    )


@lru_cache(maxsize=4)
def _cached_encoder(model_name: str):
    from genai_eval.classification.retrieval.embedding_index import load_encoder

    return load_encoder(model_name)


def _weighted_department_vote(hits: list, departments: Iterable[str]) -> tuple[str, float]:
    by_normalized = {str(value).casefold(): str(value) for value in departments}
    totals: dict[str, float] = {}
    best_similarity: dict[str, float] = {}
    for rank, hit in enumerate(hits, start=1):
        department = by_normalized.get(str(hit.queue_label).casefold())
        if not department:
            continue
        weight = max(float(hit.similarity), 0.0)
        if weight == 0:
            weight = 1e-9 / rank
        totals[department] = totals.get(department, 0.0) + weight
        best_similarity[department] = max(
            best_similarity.get(department, float("-inf")),
            float(hit.similarity),
        )
    if not totals:
        raise ValueError("No retrieved examples match the active allowed departments")
    winner = max(totals, key=lambda value: (totals[value], best_similarity[value], value))
    total_weight = sum(totals.values())
    confidence = totals[winner] / total_weight if total_weight else 0.0
    return winner, confidence


def route_ticket_with_embeddings(
    ticket_text: str,
    departments: list[str],
    top_k: int = 5,
    ticket_id: str | None = None,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> dict:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    status = embedding_routing_status(departments, index_dir=index_dir)
    if not status.available:
        raise RuntimeError(status.message)

    from genai_eval.classification.retrieval.embedding_index import (
        EmbeddingIndex,
        encode_texts,
    )

    index = EmbeddingIndex(index_dir)
    vector = encode_texts(
        _cached_encoder(index.encoder_model),
        [ticket_text],
        batch_size=1,
        show_progress_bar=False,
    )[0]
    candidates = index.query_by_vector(
        vector,
        top_k=max(top_k * 5, top_k),
        exclude_ticket_id=ticket_id,
        exclude_text=ticket_text,
    )
    allowed = {department.casefold() for department in departments}
    hits = [hit for hit in candidates if hit.queue_label.casefold() in allowed][:top_k]
    department, confidence = _weighted_department_vote(hits, departments)
    supporting_ids = [hit.ticket_id for hit in hits if hit.queue_label.casefold() == department.casefold()]
    return {
        "department": department,
        "reason": (
            "Weighted nearest-neighbor vote from reviewed tickets"
            + (f": {', '.join(supporting_ids)}" if supporting_ids else "")
        ),
        "evidence": "",
        "evidence_valid": False,
        "classifier": "embedding_nearest_neighbor",
        "model": f"embedding:{index.encoder_model}",
        "confidence": round(confidence, 6),
        "retrieval": {
            "encoder_model": index.encoder_model,
            "index_source_sha256": index.manifest["source"]["sha256"],
            "hits": [
                {
                    "ticket_id": hit.ticket_id,
                    "department": hit.queue_label,
                    "similarity": round(hit.similarity, 6),
                }
                for hit in hits
            ],
        },
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "usage_reported": True,
    }


def build_routing_index(
    rows: Iterable[dict],
    id_column: str,
    text_column: str,
    department_column: str,
    allowed_departments: list[str],
    index_dir: Path = DEFAULT_INDEX_DIR,
    reference_path: Path = DEFAULT_REFERENCE_PATH,
    encoder_model: str = DEFAULT_ENCODER,
) -> dict:
    if not _embedding_dependencies_available():
        raise RuntimeError(
            "Embedding dependencies are unavailable. Use the runtime-embeddings Docker target."
        )
    by_normalized = {department.casefold(): department for department in allowed_departments}
    records = []
    seen_ids = set()
    for row_number, row in enumerate(rows, start=1):
        ticket_id = str(row.get(id_column, "")).strip()
        text = str(row.get(text_column, "")).strip()
        requested_department = str(row.get(department_column, "")).strip()
        department = by_normalized.get(requested_department.casefold())
        if not ticket_id or not text or not requested_department:
            raise ValueError(f"Reference row {row_number} has an empty ID, text, or department")
        if not department:
            raise ValueError(
                f"Reference row {row_number} uses an unconfigured department: "
                f"{requested_department!r}"
            )
        if ticket_id in seen_ids:
            raise ValueError(f"Duplicate reference ticket ID: {ticket_id!r}")
        seen_ids.add(ticket_id)
        records.append(
            {
                "ticket_id": ticket_id,
                "text": text,
                "type": "Routing",
                "queue": department,
                "tags": [],
            }
        )
    if not records:
        raise ValueError("The reference CSV has no usable rows")

    reference_path.parent.mkdir(parents=True, exist_ok=True)
    with reference_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ticket_id", "text", "gold_type", "gold_queue", "gold_tags"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "ticket_id": record["ticket_id"],
                    "text": record["text"],
                    "gold_type": record["type"],
                    "gold_queue": record["queue"],
                    "gold_tags": "[]",
                }
            )

    from genai_eval.classification.retrieval.embedding_index import (
        encode_texts,
        write_index,
    )

    embeddings = encode_texts(
        _cached_encoder(encoder_model),
        [record["text"] for record in records],
        batch_size=32,
        show_progress_bar=False,
    )
    manifest = write_index(
        index_dir,
        records,
        embeddings,
        encoder_model,
        reference_path,
        {
            "ticket_id": "ticket_id",
            "text": "text",
            "gold_type": "gold_type",
            "gold_queue": "gold_queue",
            "gold_tags": "gold_tags",
        },
    )
    (index_dir / "allowed_labels.json").write_text(
        json.dumps(
            {
                "types": ["Routing"],
                "queues": sorted({record["queue"] for record in records}),
                "tags": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
