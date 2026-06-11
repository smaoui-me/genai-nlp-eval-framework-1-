This folder is the canonical place for classification-specific retrieval code.

Use it for:
- embedding-based candidate retrieval
- RAG-style support for ticket classification
- classification-only retrieval helpers

Keep generic reusable infrastructure in `src/genai_eval/`.
Keep future true extraction retrieval logic under `src/genai_eval/extraction/` when needed.

