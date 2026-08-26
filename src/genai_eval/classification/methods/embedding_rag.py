"""Retrieval-augmented classification using reviewed examples and an LLM."""

from __future__ import annotations

import json
from pathlib import Path

from genai_eval.classification.methods.base import EvidenceClassificationMethod
from genai_eval.classification.retrieval.embedding_index import EmbeddingIndex, encode_texts, load_encoder
from genai_eval.config_loader import load_config
from genai_eval.json_utils import parse_json_response
from genai_eval.llm_client import call_llm, get_model_name
from genai_eval.prompts import format_prompt, load_prompt_template


DEFAULT_CONFIG_PATH = Path("configs/classification/embedding_rag.yaml")


def _evidence_excerpt(text: str, maximum: int = 180) -> str:
    return str(text).strip()[:maximum]


class EmbeddingRagEvidenceClassification(EvidenceClassificationMethod):
    """Select semantically similar reviewed tickets as dynamic few-shot examples."""

    name = "embedding_rag"

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config = load_config(config_path)
        self.prompt_template = load_prompt_template(Path(self.config["prompt_path"]))
        self.llm_params = self.config.get("llm", {})
        self.top_k_tags = self.config.get("extraction", {}).get("top_k_tags")
        retrieval = self.config.get("retrieval", {})
        self.top_k = int(retrieval.get("top_k", 3))
        if self.top_k < 1:
            raise ValueError("retrieval.top_k must be at least 1")
        self.index = EmbeddingIndex(Path(retrieval["index_dir"]))
        configured_model = str(retrieval.get("encoder_model", self.index.encoder_model))
        if configured_model != self.index.encoder_model:
            raise ValueError(
                "Configured encoder does not match the index manifest: "
                f"{configured_model!r} != {self.index.encoder_model!r}"
            )
        self.encoder_model = configured_model
        self._encoder = None
        self.model_name = get_model_name()

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = load_encoder(self.encoder_model)
        return self._encoder

    def _select_hits(self, text: str, allowed_labels: dict, ticket_id: str | None):
        vector = encode_texts(self.encoder, [text], batch_size=1, show_progress_bar=False)[0]
        candidates = self.index.query_by_vector(
            vector,
            top_k=max(self.top_k * 5, self.top_k),
            exclude_ticket_id=ticket_id,
            exclude_text=text,
        )
        allowed_types = set(allowed_labels.get("types", []))
        allowed_queues = set(allowed_labels.get("queues", []))
        hits = [
            hit
            for hit in candidates
            if hit.type_label in allowed_types and hit.queue_label in allowed_queues
        ][: self.top_k]
        if not hits:
            raise ValueError("No retrieved reference examples match the active type and queue schema")
        return hits

    @staticmethod
    def _format_examples(hits, candidate_tags: list[str]) -> str:
        candidate_set = set(candidate_tags)
        sections = []
        for number, hit in enumerate(hits, start=1):
            evidence = _evidence_excerpt(hit.text)
            output = {
                "type": {"label": hit.type_label, "evidence": evidence},
                "queue": {"label": hit.queue_label, "evidence": evidence},
                "tags": [
                    {"label": tag, "evidence": evidence}
                    for tag in hit.tags
                    if not candidate_set or tag in candidate_set
                ],
            }
            sections.append(
                f"Example {number} (reference ID {hit.ticket_id}):\n"
                f"Ticket:\n---\n{hit.text}\n---\n"
                f"Output:\n{json.dumps(output, ensure_ascii=False)}"
            )
        return "\n\n".join(sections)

    def extract_record(self, text: str, allowed_labels: dict, context: dict | None = None) -> dict:
        context = context or {}
        candidate_tags = context.get("candidate_tags", allowed_labels.get("tags", []))
        ticket_id = str(context["ticket_id"]) if context.get("ticket_id") is not None else None
        hits = self._select_hits(text, allowed_labels, ticket_id)
        prompt = format_prompt(
            self.prompt_template,
            ticket_text=text,
            allowed_types=allowed_labels["types"],
            allowed_queues=allowed_labels["queues"],
            candidate_tags=candidate_tags,
            retrieved_examples=self._format_examples(hits, candidate_tags),
        )
        raw_response = call_llm(prompt, **self.llm_params)
        parsed_output, json_valid, json_error = parse_json_response(raw_response)
        validated_output, validation = self.validate_against_labels(
            prediction=parsed_output,
            allowed_labels=allowed_labels,
            candidate_tags=candidate_tags,
            top_k_tags=self.top_k_tags,
        )
        return {
            "raw_responses": {"step_single": raw_response},
            "parsed_output": parsed_output if isinstance(parsed_output, dict) else {},
            "validated_output": validated_output,
            "json_validity": {
                "step_single_json_valid": json_valid,
                "step_single_json_error": json_error,
                "all_json_valid": json_valid,
            },
            "validation": validation,
            "retrieval": {
                "encoder_model": self.encoder_model,
                "index_source_sha256": self.index.manifest["source"]["sha256"],
                "hits": [
                    {"ticket_id": hit.ticket_id, "similarity": round(hit.similarity, 6)}
                    for hit in hits
                ],
            },
        }

    def extract(self, text: str, allowed_labels: dict, context: dict | None = None) -> dict:
        return self.extract_record(text, allowed_labels, context=context)["validated_output"]
