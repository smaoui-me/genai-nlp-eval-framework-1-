# GENAI_NLP_EVAL_FRAMEWORK_1

This repository is a reusable evaluation framework for GenAI-based NLP tasks.

## Current status

- `classification/`
  The only implemented pipeline today. It predicts existing ticket labels:
  `type`, `queue`, and `tags`.
- `extraction/`
  Placeholder for future true extraction tasks, likely NER-style.
- `annotation/`
  Placeholder for future LLM-assisted pre-annotation workflows.

## Classification data

The active classification dataset lives under `data/classification/processed/`.

- `data/classification/processed/ticket_allowed_labels.json`
  Allowed label space for `types`, `queues`, and `tags`
- `data/classification/processed/ticket_extraction_eval.csv`
  Evaluation dataset for the current classification task

Note:
- `ticket_extraction_eval.csv` keeps a legacy filename for safety
- the implemented task is still classification, not extraction

## Classification pipeline

Implemented variants currently live under the classification task area:

- `zero_shot`
  One prompt predicts `type`, `queue`, and `tags`
- `agent_two_step`
  Step 1 extracts evidence from the ticket text, step 2 maps to labels
- `few_shot`
  Same prediction target with in-context examples

All active classification code should live in:

- `configs/classification/`
- `prompts/classification/`
- `scripts/classification/`
- `src/genai_eval/classification/`
- `results/classification/`

Classification-specific retrieval or embedding/RAG logic should live in:

- `src/genai_eval/classification/retrieval/`

## Evaluation

The current classification evaluation compares predictions against gold labels for:

- `type`
- `queue`
- `tags`

Reported metrics include:

- `type_accuracy`
- `type_macro_f1`
- `queue_accuracy`
- `queue_macro_f1`
- `tag_micro_precision`
- `tag_micro_recall`
- `tag_micro_f1`
- `tag_row_precision`
- `tag_row_recall`
- `tag_row_f1`
- `invalid_json_rate`
- `invalid_label_rate`
- `evidence_valid_rate`

Classification outputs and evaluation artifacts should be written under:

- `results/classification/`
- `results/classification/evaluation/`

## Run commands

Run one-row classification checks:

- `python scripts/classification/run_ticket_classification_with_evidence.py --method zero_shot --limit 1`
- `python scripts/classification/run_ticket_classification_with_evidence.py --method agent_two_step --limit 1`
- `python scripts/classification/run_ticket_classification_with_evidence.py --method few_shot --limit 1`

Compare saved classification runs:

- `python scripts/classification/compare_ticket_classification_with_evidence.py`
