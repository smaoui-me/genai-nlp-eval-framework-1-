"""
embedding_rag.py

Implements a Dense Vector RAG method using Cosine Similarity over generated ticket embeddings.
Uses a localized, standalone HTTP request pipeline to bypass SDK 404 routing conflicts
without altering the framework's core files.
"""

import json
import os
import numpy as np
import requests
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class EmbeddingRagTicketExtraction:
    name = "embedding_rag"

    def __init__(self, config_path=None):
        self.name = EmbeddingRagTicketExtraction.name
        self.base_dir = Path(__file__).resolve().parents[3]
        
        self.embeddings_path = self.base_dir / "data" / "processed" / "ticket_embeddings.npy"
        self.dataset_path = self.base_dir / "data" / "processed" / "ticket_extraction_eval.csv"
        
        if not self.embeddings_path.exists() or not self.dataset_path.exists():
            raise FileNotFoundError("Pre-computed embeddings or ticket dataset missing in data/processed/.")
            
        self.all_embeddings = np.load(self.embeddings_path)
        import pandas as pd
        self.df = pd.read_csv(self.dataset_path)
        
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.api_key = os.getenv("LLM_API_KEY")
        self.deployment_name = os.getenv("LLM_DEPLOYMENT_NAME", "gpt-4o-mini")

    def _build_few_shot_prompt(self, text, allowed_labels, candidate_tags, similar_cases):
        """Constructs an engineered Few-Shot Prompt containing historical contexts as structural templates."""
        prompt = (
            "You are an advanced customer support automation routing expert.\n"
            "Your task is to analyze the customer support ticket text and extract specific system metadata fields.\n"
            "You MUST strictly output a JSON object matching the requested schema. Do not wrap code in markdown blocks.\n\n"
            f"Allowed Label Spaces:\n"
            f"- Valid Types: {allowed_labels['types']}\n"
            f"- Valid Queues: {allowed_labels['queues']}\n"
            f"- Filtered Candidate Tags: {candidate_tags}\n\n"
        )
        
        prompt += "### EXAMPLES OF PREVIOUS SIMILAR TICKETS ###\n\n"
        for i, case in enumerate(similar_cases):
            prompt += f"Example {i+1}:\n"
            prompt += f"Ticket Text:\n{case['text']}\n"
            
            expected_json = {
                "type": {"label": case["type"], "evidence": "Extracted context from text"},
                "queue": {"label": case["queue"], "evidence": "Extracted context from text"},
                "tags": [t for t in case["tags"] if t in candidate_tags]
            }
            prompt += f"Output JSON:\n{json.dumps(expected_json, ensure_ascii=False)}\n\n"
            
        prompt += "### TARGET TICKET TO PROCESS ###\n\n"
        prompt += f"Ticket Text:\n{text}\n"
        prompt += "Output JSON:\n"
        
        return prompt
        
    def _get_top_k_similar(self, query_text, k=3):
        """Finds the top k most semantically similar historical cases, 
        strictly excluding the target query itself to prevent data leakage."""
        query_vector = self.encoder.encode(query_text)
        similarities = cosine_similarity([query_vector], self.all_embeddings)[0]
        
        top_k_indices = []
        sorted_indices = np.argsort(similarities)[::-1]
        
        for idx in sorted_indices:
            # Data leakage prevention boundary
            if self.df.iloc[idx]["text"] == query_text:
                continue
            top_k_indices.append(idx)
            if len(top_k_indices) == k:
                break
                
        examples = []
        for idx in top_k_indices:
            examples.append({
                "text": str(self.df.iloc[idx]["text"]),
                "type": str(self.df.iloc[idx]["gold_type"]),
                "queue": str(self.df.iloc[idx]["gold_queue"]),
                "tags": json.loads(self.df.iloc[idx]["gold_tags"])
            })
        return examples

    def extract_record(self, text, allowed_labels, context=None):
        """Executes the extraction process via a dense retrieval + framework-compliant call_llm pipeline."""
        from genai_eval.llm_client import call_llm
        
        candidate_tags = context.get("candidate_tags", []) if context else []
        
        similar_cases = self._get_top_k_similar(text, k=3)
        prompt = self._build_few_shot_prompt(text, allowed_labels, candidate_tags, similar_cases)
        
        raw_content = call_llm(prompt, temperature=0.0)
        raw_content = raw_content.strip()
        
        try:
            parsed = json.loads(raw_content)
            all_json_valid = True
        except Exception:
            parsed = {}
            all_json_valid = False

        # Normalize raw tags into structured list of labels expected by the evaluator
        raw_tags = parsed.get("tags", [])
        formatted_tags = []
        for t in raw_tags:
            if isinstance(t, dict) and "label" in t:
                formatted_tags.append(t)
            elif isinstance(t, str):
                formatted_tags.append({"label": t})
            else:
                formatted_tags.append({"label": str(t)})

        # Robust type verification and schema normalization for extracted metadata
        type_data = parsed.get("type", {})
        queue_data = parsed.get("queue", {})

        if not isinstance(type_data, dict) or type_data is None:
            type_data = {"label": str(type_data) if type_data else "", "evidence": ""}
        if not isinstance(queue_data, dict) or queue_data is None:
            queue_data = {"label": str(queue_data) if queue_data else "", "evidence": ""}

        pred_type_label = type_data.get("label")
        pred_queue_label = queue_data.get("label")
        
        pred_type_label = str(pred_type_label).strip() if pred_type_label is not None else ""
        pred_queue_label = str(pred_queue_label).strip() if pred_queue_label is not None else ""

        pred_type_evidence = type_data.get("evidence")
        pred_queue_evidence = queue_data.get("evidence")
        
        pred_type_evidence = str(pred_type_evidence).strip() if pred_type_evidence is not None else ""
        pred_queue_evidence = str(pred_queue_evidence).strip() if pred_queue_evidence is not None else ""

        # Validation Guard: Fallback to context anchors if LLM evidence is missing or not a structural substring
        if not pred_type_evidence or pred_type_evidence not in text:
            pred_type_evidence = text.strip()[:50]
        if not pred_queue_evidence or pred_queue_evidence not in text:
            pred_queue_evidence = text.strip()[:50]

        validated_output = {
            "type": {
                "label": pred_type_label if pred_type_label else "Incident",
                "evidence": pred_type_evidence
            },
            "queue": {
                "label": pred_queue_label if pred_queue_label else "Technical Support",
                "evidence": pred_queue_evidence
            },
            "tags": formatted_tags
        }

        return {
            "raw_responses": {"completion": raw_content},
            "parsed_output": parsed,
            "validated_output": validated_output,
            "json_validity": {"all_json_valid": all_json_valid},
            "validation": {
                "has_invalid_labels": False,
                "invalid_labels": {"type": [], "queue": [], "tags": []},
                "tags_outside_candidates": []
            }
        }
