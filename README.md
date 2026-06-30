# GENAI_NLP_EVAL_FRAMEWORK_1

This repository is a reusable evaluation framework for GenAI-based NLP tasks.

## Current status

- `classification/`
  Implemented ticket label prediction pipeline. It predicts existing ticket labels:
  `type`, `queue`, and `tags`.
- `extraction/`
  Implemented location-only NER-style extraction pipeline on FewNERD.
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

The extraction evaluation compares predicted location spans against gold spans on FewNERD.

Reported extraction metrics include:

- `strict_precision`
- `strict_recall`
- `strict_f1`
- `lenient_precision`
- `lenient_recall`
- `lenient_f1`
- `json_valid_rate`
- `invalid_json_rate`
- `invalid_label_rate`

Extraction outputs and evaluation artifacts should be written under:

- `results/extraction/`
- `results/extraction/evaluation/`

## Current results

Latest 100-row classification results:

- `zero_shot`
  - `type_accuracy = 0.84`
  - `queue_accuracy = 0.46`
  - `tag_micro_f1 = 0.2695`
  - `tag_row_f1 = 0.2679`
  - `evidence_valid_rate = 0.9567`
- `agent_two_step`
  - `type_accuracy = 0.80`
  - `queue_accuracy = 0.46`
  - `tag_micro_f1 = 0.2453`
  - `tag_row_f1 = 0.2471`
  - `evidence_valid_rate = 0.9455`

Interpretation:

- `zero_shot` currently performs better than `agent_two_step` on `type` and `tags`
- both methods are similar on `queue`
- both methods are stable structurally with `invalid_json_rate = 0.0` and `invalid_label_rate = 0.0`

Latest 100-row extraction results on location-only FewNERD:

- `zero_shot_structured`
  - `strict_precision = 0.7608`
  - `strict_recall = 0.8503`
  - `strict_f1 = 0.8030`
  - `json_valid_rate = 1.0`
- `zero_shot_freeform`
  - `strict_precision = 0.7667`
  - `strict_recall = 0.8610`
  - `strict_f1 = 0.8111`
  - `json_valid_rate = 0.0` because JSON is not used for this method

Interpretation:

- both extraction methods perform strongly on location span extraction
- `freeform` is slightly better than `structured` on this 100-row slice
- `strict` and `lenient` scores are identical here, which means text extraction and index assignment are aligned on the accepted predictions
- both extraction methods are stable after switching prompts to explicit token-index supervision

## Run commands

Run one-row classification checks:

- `python scripts/classification/run_ticket_classification_with_evidence.py --method zero_shot --limit 1`
- `python scripts/classification/run_ticket_classification_with_evidence.py --method agent_two_step --limit 1`
- `python scripts/classification/run_ticket_classification_with_evidence.py --method few_shot --limit 1`

Compare saved classification runs:

- `python scripts/classification/compare_ticket_classification_with_evidence.py`

Run extraction comparisons:

- `python scripts/extraction/run_zero_shot_structured.py --limit 100`
- `python scripts/extraction/run_zero_shot_freeform.py --limit 100`
- `python scripts/extraction/compare.py`
