# Stage 1 SciREX development pilot

## Purpose

This run validates the end-to-end batch path and identifies prompt problems
before final evaluation. It is a development result and must not be presented
as held-out test performance.

## Protocol

- Dataset split: SciREX `dev`
- Selection: 20 deterministic, document-disjoint windows
- Balance: 5 source windows from each of `short`, `medium`, `long`, and `very_long`
- Unique source papers: 20
- Model: `hosted:gpt-5.4`
- Method: `few_shot_structured`
- Labels: `Method`, `Task`, `Metric`, `Material`
- Temperature: `0.0`
- Processing limit: first 10 annotation-tool sentences per window
- Matching: exact character span and case-insensitive exact label

The source windows contain 1,708 SciREX sentences in full, but this inexpensive
pilot deliberately processed only their prefixes. The annotation tool's own
sentence splitter produced 167 processed sentences/calls. Consequently, bucket
names describe the full source windows, not the amount of text processed in
this pilot. These bucket results must not be used as evidence of full
long-document performance.

## Results

All 20 examples completed. There were zero failed examples and zero invalid
structured responses.

| Metric | Value |
|---|---:|
| Strict precision | 42.44% |
| Strict recall | 57.26% |
| Strict F1 | 48.75% |
| Macro document F1 | 48.65% |
| True positives | 205 |
| False positives | 278 |
| False negatives | 153 |
| Predictions | 483 |
| Gold entities in processed prefixes | 358 |
| LLM calls | 167 |
| Wall-clock model time | 543.25 s |

### Per-label strict results

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| Material | 7.27% | 57.14% | 12.90% |
| Method | 53.39% | 51.75% | 52.56% |
| Metric | 40.00% | 66.67% | 50.00% |
| Task | 55.56% | 68.42% | 61.32% |

## Interpretation and next decision

The pipeline is mechanically stable, but the prompt is not ready for final
evaluation. It over-predicts entities, especially `Material`: 102 false
positive Material spans produced only 8 true positives. Task is currently the
strongest label. Prompt work should therefore focus on SciREX-specific label
definitions and negative examples, particularly the distinction between a
research material/dataset and generic technical nouns.

After changing the prompt, rerun this exact development manifest and compare
the saved metrics. Freeze the prompt only after development performance and
error inspection are acceptable. Then run a fresh, document-disjoint sample
from `test`; do not tune on test results.

## Reproduction

```powershell
python scripts/annotation/select_scirex_pilot.py
python scripts/annotation/run_scirex_batch.py --method few_shot_structured --max-sentences 10
python scripts/annotation/evaluate_scirex_batch.py
```

The selection manifest is `eval/corpora/scirex_dev_pilot_20.jsonl`. Runtime
artifacts are written beneath `results/annotation/scirex_dev_pilot_20/` and
include aggregate metrics, per-example metrics, false positives, false
negatives, and failures. The runner is resumable: completed example IDs are
skipped on subsequent invocations.
