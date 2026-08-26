# Local retrieval artifacts

This directory is reserved for user-owned, reviewed classification examples
and their generated embedding index. Its contents are ignored by Git except
for this file because they may contain confidential tickets.

The default layout after indexing is:

```text
data/classification/retrieval/
├── reference_tickets.csv     # user-provided and reviewed; never the test set
└── index/
    ├── allowed_labels.json
    ├── embeddings.npy
    ├── manifest.json
    └── records.jsonl
```

The manifest records the source hash, encoder model, column mapping, dimensions,
and artifact hashes. Rebuild the index whenever the reference CSV or encoder
model changes. Do not commit these artifacts unless the data is explicitly
approved for publication.
