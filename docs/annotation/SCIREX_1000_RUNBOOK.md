# SciREX 100/1,000-example study runbook

## Readiness gates

The 1,000-window run is permitted only after all of these pass:

1. Unit and artifact tests pass.
2. The network-free readiness checker passes.
3. `--dry-run` reports the expected manifest and prompt configuration.
4. A controlled interruption test proves sentence-level resume.
5. The 100-example operational run completes without systematic API failures.
6. Provider prices are entered if a monetary spend ceiling is required.
7. The operational results are reviewed before launching 1,000 windows.

## Fixed study configuration

- Prompt: UI-equivalent schema suggestion
- Labels: Method, Task, Metric, Dataset (`Dataset` maps to SciREX `Material`)
- Model: configured hosted model; record its exact deployment ID
- Temperature: 0
- Coverage: complete windows
- Workers: 4
- Global request ceiling: 120 logical calls/minute
- Checkpoint: atomic state after every successful sentence and before each call
- Metrics: strict, +/-1-start and +/-2-end token tolerance, and same-label overlap
- Primary accuracy: official test split strict F1
- Confidence intervals: resample by `doc_id`, retaining all windows from a paper

The 100-example operational manifest is balanced by length, contains 100
different **training-split** papers, and is only a reliability/cost gate. It
does not expose held-out test papers. Its exact call and mention counts are
printed by the dry run and artifact validator.

The complete benchmark has 1,000 windows from 438 papers, 85,220 SciREX source
sentences, 62,145 exact application-level calls, and 109,138 gold mentions.
Windows can overlap; they are not 1,000 statistically independent papers.

## Pricing safety

The committed YAML files intentionally set token prices and `max_cost_usd` to
zero because the repository cannot know the user's provider contract. Before a
paid run, obtain the actual input/output prices for the configured deployment
and set all three values. When a positive spend limit is configured, the runner
stops safely if the endpoint fails to report token usage. Parallel workers can
overshoot a newly reached monetary threshold by at most their already active
calls; the call budget is reserved before requests and cannot be exceeded.

## Network-free preflight

```powershell
python scripts/annotation/check_scirex_study_readiness.py
```

This verifies counts, document isolation, offsets, complete-window settings,
exact application call budgets, and manifest hashes. Pricing warnings remain
until the actual provider rates and approved spend ceiling are entered.

## Dry runs

```powershell
python scripts/annotation/run_scirex_batch.py `
  --config configs/annotation/scirex_operational_100.yaml --dry-run

python scripts/annotation/run_scirex_batch.py `
  --config configs/annotation/scirex_full_1000.yaml --dry-run
```

Expected exact call counts are 5,868 and 62,145 respectively.
At the configured 120 logical calls/minute ceiling, the theoretical minimum
times are about 49 minutes and 8 hours 38 minutes. Provider latency, retries,
and throttling can make the real runs longer.

## Operational run (required first)

```powershell
python scripts/annotation/run_scirex_batch.py `
  --config configs/annotation/scirex_operational_100.yaml

python scripts/annotation/evaluate_scirex_batch.py `
  --manifest eval/corpora/scirex_operational_100.jsonl `
  --predictions results/annotation/scirex_operational_100/predictions.jsonl `
  --output-dir results/annotation/scirex_operational_100/evaluation
```

Inspect failures, missing usage, invalid JSON, per-split results, cost, and
length/label tables. Do not proceed if errors are systematic or cost exceeds
the approved estimate.

## Full run

```powershell
python scripts/annotation/run_scirex_batch.py `
  --config configs/annotation/scirex_full_1000.yaml

python scripts/annotation/evaluate_scirex_batch.py `
  --manifest data/annotation/processed/scirex/examples_1000.jsonl `
  --predictions results/annotation/scirex_full_1000/predictions.jsonl `
  --output-dir results/annotation/scirex_full_1000/evaluation
```

Ctrl+C is safe between sentence calls. Rerunning the same command reads each
document checkpoint and continues from its next unfinished sentence. The
runner rejects prompt/model/configuration mixing in an existing output.

## Reporting rules

Report train, development, and test metrics separately. The official test split
is the primary accuracy result. The combined 1,000-window score is descriptive
load/error analysis only. State 438 unique papers, overlapping windows, exact
call count, provider token usage, estimated spend, failures, prompt hash, and
clustered confidence intervals. Preserve the ignored runtime result directory
in controlled artifact storage before cleaning the workspace.

Use `SCIREX_STUDY_REPORT_TEMPLATE.md` for the final write-up and do not alter
the prompt after reading the official test result.
