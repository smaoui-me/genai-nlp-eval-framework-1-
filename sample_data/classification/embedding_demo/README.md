# Synthetic embedding-retrieval demo data

These tickets are synthetic and safe to use for a functional demonstration.
They are intentionally small and are not evidence of production accuracy.

- reference_tickets.csv: 20 reviewed examples used to build the index.
- development_tickets.csv: 8 labeled examples for prompt and top-k choices.
- held_out_tickets.csv: 12 untouched labeled examples for final comparison.
- unlabeled_tickets.csv: 4 examples for prediction-only operation.

Ticket IDs and exact texts are disjoint across the four files. The held-out and
development labels are all represented in the reference file. Never add either
labeled evaluation file to the embedding index.

See docs/TESTING.md for executable commands.
