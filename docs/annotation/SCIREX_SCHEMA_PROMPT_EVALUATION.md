# Editable schema prompt and flexible-span evaluation

## Product change

The annotation application previously used static prompt files that users
could neither inspect nor modify. It now creates a deterministic suggestion
from the selected dataset and labels, displays the complete template in the UI,
and lets the user edit or restore it before making an LLM call. Custom labels
receive a generic definition in the suggestion and can be refined directly in
the editor. Every saved run records the final prompt SHA-256.

For SciREX, the confusing UI label `Material` is now displayed as `Dataset`.
Evaluation maps both names to the same canonical class, preserving compatibility
with the released gold annotations.

## Evaluation modes

- **Strict:** exact character boundaries and label.
- **Boundary tolerant:** same label, start within one token and end within two
  tokens, using one-to-one matching.
- **Overlap:** any same-label character overlap, using one-to-one matching.

Strict remains the primary research result. The other metrics quantify whether
the pre-annotation directed a reviewer to the useful phrase.

## Development result

On the fixed 20-paper development pilot (first ten app-detected sentences per
window), the schema-derived suggestion outperformed the previous static prompt:

| Prompt | Strict precision | Strict recall | Strict F1 |
|---|---:|---:|---:|
| Previous static few-shot | 42.44% | 57.26% | 48.75% |
| Editable schema suggestion | **65.66%** | **60.89%** | **63.19%** |

The suggested prompt achieved 70.14% boundary-tolerant F1 and 74.20% overlap
F1. This result was used to select the new prompt behavior.

## Fresh confirmation protocol

The earlier 20 test papers had already informed error analysis, so they were
excluded. A new deterministic sample used 20 different official test papers,
five per source-length bucket. The complete windows contained 1,633 SciREX
source sentences and 2,359 gold mentions.

- Model: `hosted:gpt-5.4`
- Method/parser: `few_shot_structured`
- Prompt mode: editable schema suggestion
- User-facing labels: Method, Task, Metric, Dataset
- Temperature: 0
- Full-window processing
- Actual LLM calls: 1,224
- Failed documents: 0
- Invalid structured responses: 0

## Fresh confirmation results

| Metric | Precision | Recall | F1 |
|---|---:|---:|---:|
| Strict | 59.59% | 55.32% | **57.38%** |
| ±1 start / ±2 end tokens | 69.95% | 64.94% | **67.36%** |
| Same-label overlap | 73.01% | 67.78% | **70.30%** |

The paper-level bootstrap 95% interval for strict F1 was 53.56–60.73%.

### Strict results by label

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| Dataset | 50.50% | 66.45% | 57.39% |
| Method | 58.96% | 60.72% | 59.83% |
| Metric | 56.40% | 42.92% | 48.74% |
| Task | 69.46% | 42.96% | 53.09% |

The Dataset rename plus narrower definition resolved the previous catastrophic
Material behavior: its strict F1 rose from 8.72% in the superseded baseline to
57.39% on the fresh confirmation sample. Because the samples differ, this is
evidence of a large improvement, not a paired per-paper comparison.

## Interpretation

The prompt generator is deterministic rather than another LLM call: it derives
definitions from the schema, costs nothing, and remains reproducible. A user can
edit it for a new domain. Future learning from corrections should first retrieve
approved examples into prompts; fine-tuning should wait until the project has a
large, consistently reviewed corpus.

The results support two distinct statements: exact benchmark reproduction is
57.38% F1, while useful phrase localization reaches 67.36–70.30% F1 under the
predeclared flexible policies. Human review remains required.

## Reproduction

```powershell
python scripts/annotation/select_scirex_pilot.py `
  --input data/annotation/processed/scirex/test.jsonl `
  --output eval/corpora/scirex_test_confirmation_20.jsonl `
  --split test --per-bucket 5 `
  --exclude-manifest eval/corpora/scirex_test_20.jsonl

python scripts/annotation/run_scirex_batch.py `
  --input eval/corpora/scirex_test_confirmation_20.jsonl `
  --output results/annotation/scirex_test_confirmation_20_suggested/predictions.jsonl `
  --method few_shot_structured --prompt-mode suggested `
  --max-sentences 0 --workers 4

python scripts/annotation/evaluate_scirex_batch.py `
  --manifest eval/corpora/scirex_test_confirmation_20.jsonl `
  --predictions results/annotation/scirex_test_confirmation_20_suggested/predictions.jsonl `
  --output-dir results/annotation/scirex_test_confirmation_20_suggested/evaluation
```
