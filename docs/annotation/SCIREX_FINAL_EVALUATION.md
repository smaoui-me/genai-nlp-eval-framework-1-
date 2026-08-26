# SciREX annotation pre-labeling evaluation

> Baseline/superseded result. The later editable schema-suggested prompt and a
> fresh disjoint confirmation sample are documented in
> `SCIREX_SCHEMA_PROMPT_EVALUATION.md`.

## Research question

How accurately does the LLM pre-annotation component identify SciREX Method,
Task, Metric, and Material mentions, including in long scientific texts?

SciREX is a document-level scientific information-extraction dataset introduced
by Jain et al. (ACL 2020). Its entity-recognition task identifies mentions of
Method, Task, Metric, and Dataset; the released files use `Material` for the
dataset/data-resource label. Source: <https://doi.org/10.18653/v1/2020.acl-main.670>.

## Leakage controls and model selection

Prompt work used only the SciREX development split. A deterministic pilot
selected 20 windows from 20 unique development papers, balanced across the four
length buckets. Three configurations were compared on identical text prefixes:

| Development configuration | Precision | Recall | Strict F1 |
|---|---:|---:|---:|
| Generic few-shot structured | 42.44% | 57.26% | **48.75%** |
| SciREX-specific conservative | 49.07% | 22.07% | 30.44% |
| SciREX-specific exhaustive | 48.61% | 19.55% | 27.89% |

The generic few-shot structured prompt was frozen because it had the highest
development F1. No prompt or parameter was changed after observing test output.

## Held-out test protocol

- Split: official SciREX `test`
- Sampling: deterministic, 5 windows per length bucket
- Sample size: 20 full windows from 20 unique papers
- Full source size: 1,642 SciREX sentences and 2,184 gold entities
- Model: `hosted:gpt-5.4`
- Method: `few_shot_structured`
- Temperature: `0.0`
- Coverage: complete selected windows, without a sentence cap
- Matching: exact window-relative character span and case-insensitive exact label
- Confidence interval: 2,000-iteration percentile bootstrap over papers, seed 42

The application sentence splitter merged some SciREX source boundaries and made
1,097 LLM calls. This does not remove text: complete-window character coverage
was retained and gold comparison used the exact processed character boundary.

## Held-out results

All 20 papers completed, with zero API failures and zero invalid JSON responses.

| Metric | Result |
|---|---:|
| Strict precision | 38.14% |
| Strict recall | 53.98% |
| Strict micro F1 | 44.70% |
| Macro paper F1 | 44.39% |
| True positives | 1,179 |
| False positives | 1,912 |
| False negatives | 1,005 |
| Predictions | 3,091 |
| Gold entities | 2,184 |
| LLM calls | 1,097 |
| Invalid structured responses | 0 |

Paper-level bootstrap 95% intervals were precision 34.58–42.65%, recall
50.66–58.58%, and strict F1 41.25–49.04%.

### Per-label strict results

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| Method | 54.80% | 54.16% | 54.48% |
| Task | 41.75% | 51.16% | 45.98% |
| Metric | 30.25% | 55.19% | 39.08% |
| Material | 4.65% | 68.63% | 8.72% |

### By source-length bucket

| Bucket | Precision | Recall | F1 |
|---|---:|---:|---:|
| Short | 38.24% | 56.52% | 45.61% |
| Medium | 46.67% | 65.33% | 54.44% |
| Long | 42.19% | 56.08% | 48.15% |
| Very long | 34.04% | 50.26% | 40.59% |

The intervals are also stored in the generated `summary.json` so they remain
tied to the exact prediction artifact.

## Interpretation

The system is mechanically reliable but not accurate enough for unattended
annotation. It found slightly more than half of the gold entities, while only
38% of its predictions matched the dataset exactly. The principal failure is
Material over-prediction: 717 false positives versus 35 true positives.

Performance degrades on very-long windows (40.59% F1) compared with medium
windows (54.44% F1), but does not collapse. This supports the narrower claim
that the pipeline can technically process long documents while prediction
quality remains length-sensitive and requires human review.

These results measure strict pre-annotation accuracy, not annotator time saved.
They should be reported together with a later user study of acceptance, edit,
deletion, and annotation time. The 20-paper sample is suitable for a project
evaluation and error analysis, but the intervals and sample size must be stated;
it is not equivalent to evaluating all 66 official test papers.

## Reproduction

```powershell
python scripts/annotation/select_scirex_pilot.py `
  --input data/annotation/processed/scirex/test.jsonl `
  --output eval/corpora/scirex_test_20.jsonl `
  --split test --per-bucket 5

python scripts/annotation/run_scirex_batch.py `
  --input eval/corpora/scirex_test_20.jsonl `
  --output results/annotation/scirex_test_20_full/predictions.jsonl `
  --method few_shot_structured --max-sentences 0 --workers 4

python scripts/annotation/evaluate_scirex_batch.py `
  --manifest eval/corpora/scirex_test_20.jsonl `
  --predictions results/annotation/scirex_test_20_full/predictions.jsonl `
  --output-dir results/annotation/scirex_test_20_full/evaluation
```

The runner records the prompt SHA-256, model, method, temperature, sentence
limit, timestamps, call counts, and failures. It rejects attempts to resume into
an output file created with a different configuration.
