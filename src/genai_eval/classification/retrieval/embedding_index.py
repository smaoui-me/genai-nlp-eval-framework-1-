"""Persisted dense-vector index for labeled classification examples."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


INDEX_VERSION = 1
EMBEDDINGS_FILE = "embeddings.npy"
RECORDS_FILE = "records.jsonl"
MANIFEST_FILE = "manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def load_encoder(model_name: str):
    """Load an optional Sentence Transformers encoder with a useful error."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Embedding retrieval requires the optional dependencies. Install "
            "requirements-embeddings.txt or use the Docker `embeddings` service."
        ) from exc
    return SentenceTransformer(model_name)


def encode_texts(
    encoder,
    texts: list[str],
    batch_size: int = 32,
    show_progress_bar: bool = True,
) -> np.ndarray:
    values = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(values, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError(f"Encoder returned an invalid shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Encoder returned non-finite values")
    return embeddings


def write_index(
    output_dir: Path,
    records: list[dict],
    embeddings: np.ndarray,
    encoder_model: str,
    source_path: Path,
    columns: dict[str, str],
) -> dict:
    """Write normalized embeddings, labeled records, and a verifiable manifest."""
    if not records:
        raise ValueError("Cannot build an embedding index without records")

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(records):
        raise ValueError("Embedding rows must match record count")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(matrix).all():
        raise ValueError("Embeddings must be finite and non-zero")
    matrix = matrix / norms

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / EMBEDDINGS_FILE
    records_path = output_dir / RECORDS_FILE
    manifest_path = output_dir / MANIFEST_FILE

    np.save(embeddings_path, matrix, allow_pickle=False)
    with records_path.open("w", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "index_version": INDEX_VERSION,
        "encoder_model": encoder_model,
        "normalized_embeddings": True,
        "record_count": len(records),
        "embedding_dimension": int(matrix.shape[1]),
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "columns": columns,
        },
        "artifacts": {
            EMBEDDINGS_FILE: sha256_file(embeddings_path),
            RECORDS_FILE: sha256_file(records_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


@dataclass(frozen=True)
class RetrievalHit:
    ticket_id: str
    text: str
    type_label: str
    queue_label: str
    tags: list[str]
    similarity: float


class EmbeddingIndex:
    """Load and query an on-disk index without requiring scikit-learn."""

    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)
        manifest_path = self.index_dir / MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Embedding index not found at {self.index_dir}. "
                "Run scripts/classification/build_embedding_index.py first."
            )

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("index_version") != INDEX_VERSION:
            raise ValueError(f"Unsupported embedding index version: {self.manifest.get('index_version')}")

        embeddings_path = self.index_dir / EMBEDDINGS_FILE
        records_path = self.index_dir / RECORDS_FILE
        for path in (embeddings_path, records_path):
            expected = self.manifest.get("artifacts", {}).get(path.name)
            if not expected or sha256_file(path) != expected:
                raise ValueError(f"Embedding index artifact failed checksum validation: {path}")

        self.embeddings = np.load(embeddings_path, allow_pickle=False)
        self.records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
        expected_shape = (
            int(self.manifest["record_count"]),
            int(self.manifest["embedding_dimension"]),
        )
        if self.embeddings.shape != expected_shape or len(self.records) != expected_shape[0]:
            raise ValueError("Embedding index shape does not match its manifest")
        if not np.isfinite(self.embeddings).all():
            raise ValueError("Embedding index contains non-finite values")
        norms = np.linalg.norm(self.embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError("Embedding index rows are not normalized")

    @property
    def encoder_model(self) -> str:
        return str(self.manifest["encoder_model"])

    def query_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int,
        exclude_ticket_id: str | None = None,
        exclude_text: str | None = None,
    ) -> list[RetrievalHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.embeddings.shape[1]:
            raise ValueError("Query vector dimension does not match the embedding index")
        norm = float(np.linalg.norm(vector))
        if norm == 0 or not np.isfinite(vector).all():
            raise ValueError("Query vector must be finite and non-zero")
        similarities = self.embeddings @ (vector / norm)

        excluded_id = str(exclude_ticket_id) if exclude_ticket_id is not None else None
        excluded_text = normalize_text(exclude_text) if exclude_text else None
        hits: list[RetrievalHit] = []
        for index in np.argsort(similarities)[::-1]:
            record = self.records[int(index)]
            if excluded_id is not None and str(record["ticket_id"]) == excluded_id:
                continue
            if excluded_text and normalize_text(record["text"]) == excluded_text:
                continue
            hits.append(
                RetrievalHit(
                    ticket_id=str(record["ticket_id"]),
                    text=str(record["text"]),
                    type_label=str(record["type"]),
                    queue_label=str(record["queue"]),
                    tags=[str(tag) for tag in record.get("tags", [])],
                    similarity=float(similarities[int(index)]),
                )
            )
            if len(hits) >= top_k:
                break
        return hits
