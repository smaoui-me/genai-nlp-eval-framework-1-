# SciREX pre-annotation study report template

Use this after the run. Replace brackets with values from the committed config,
prediction artifact, and generated `summary.json`; do not manually recompute
metrics in a spreadsheet.

## Research question

How accurately and reliably does the LLM-assisted annotation system identify
Method, Task, Metric, and Dataset mentions in scientific text, and how does
performance change with document-window length?

## Protocol

- Dataset: SciREX release, normalized into 1,000 coherent same-paper windows
- Statistical unit: source paper (`doc_id`), not overlapping window
- Model/deployment: [model ID]
- Prompt SHA-256: [hash]
- Manifest SHA-256: [hash]
- Temperature: 0
- Output parser: structured JSON
- Matching: strict character span + label (primary); boundary-tolerant and
  same-label overlap (secondary)
- Confidence interval: 2,000-iteration `doc_id`-clustered percentile bootstrap
- Operational gate: 100 training papers, reviewed before the full run
- Held-out policy: no prompt changes after inspecting final test metrics

## Run integrity

Report all of the following: expected/completed/failed examples, coverage rate,
logical calls, invalid responses, missing token-usage calls, input/output tokens,
estimated cost, elapsed time, and whether the run resumed from checkpoints.
A study with missing predictions must not be presented as a complete 1,000-run.

## Primary result

The primary result is strict micro F1 on the official **test** split, with its
paper-clustered 95% interval. Also report strict precision, recall, macro-paper
F1, number of test windows, and number of unique test papers.

## Secondary results

Report, clearly labeled as secondary:

1. Boundary-tolerant and overlap precision/recall/F1 on test.
2. Strict results by entity label and length bucket.
3. Train and development results separately from test.
4. Full 1,000-window aggregate as a descriptive load result, not an independent
   1,000-paper accuracy estimate.
5. API reliability, invalid-output rate, throughput, token use, and cost.

## Error analysis

Sample false positives and false negatives by label and length bucket from the
generated CSV files. Have a reviewer categorize at least 100 errors using a
fixed taxonomy (boundary, wrong label, missed abbreviation, over-general span,
spurious generic phrase, dataset-guideline ambiguity, other). Report counts and
examples without changing the locked test prompt.

## Claims and limitations

Distinguish exact benchmark accuracy from useful pre-annotation localization.
Overlapping windows reduce independence, which is why intervals cluster by
paper. SciREX measures scientific NER and does not establish performance in
unseen domains. Accuracy does not by itself prove annotation-time savings; that
requires a separate blinded human study comparing manual annotation with
pre-annotation review.
