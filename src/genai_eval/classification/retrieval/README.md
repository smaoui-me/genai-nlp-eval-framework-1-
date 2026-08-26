# Classification retrieval

`embedding_index.py` implements the persisted reference-example index used by
the `embedding_rag` classification method. It intentionally contains no
company dataset and imports Sentence Transformers only when an encoder is
actually needed.

Retrieval data must be reviewed and separated from evaluation data. Exact
ticket IDs and normalized duplicate text are excluded at query time as a
second leakage safeguard, but this does not replace a proper train/test split.
