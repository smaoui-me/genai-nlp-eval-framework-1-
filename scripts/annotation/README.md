# SciREX annotation preprocessing

From the repository root:

```powershell
python scripts/annotation/prepare_scirex.py `
  --raw-dir data/annotation/raw/release_data `
  --output-dir data/annotation/processed/scirex `
  --config configs/annotation/scirex_preprocessing.yaml

python scripts/annotation/validate_scirex.py `
  --processed-dir data/annotation/processed/scirex
```

Use `inspect_scirex.py` for readable previews of processed examples. The raw
release is never downloaded by these scripts.

## Clean CSV exports

The normalized 1,000-example benchmark can be exported into three linked CSV
tables:

```powershell
python scripts/annotation/export_scirex_csv.py
```

The files are written under `data/annotation/processed/scirex/csv/`:

- `examples_1000.csv`: one row per benchmark window, including its complete text.
- `sentences_1000.csv`: one row per sentence in a window.
- `entities_1000.csv`: one row per SciREX gold annotation.

On a public clone, reconstruct the ignored runnable JSONL from these linked
tables with:

```powershell
python scripts/annotation/rebuild_scirex_jsonl_from_csv.py
```

The CSV export omits section spans because neither the annotation runner nor
the evaluator consumes them. The reconstructed JSONL therefore has empty
`sections` arrays and a different file hash from the canonical artifact made
directly from the raw SciREX release.

Join the tables using `example_id`. Use `doc_id` to identify rows that came
from the same original scientific paper.

## How documents, windows, and sentences connect

One raw SciREX JSONL row is one complete scientific document. Preprocessing
keeps its original sentence order and assigns every sentence a
`source_sentence_index` inside that document.

The 1,000 evaluation examples are contiguous windows cut from those documents;
sentences are never randomly concatenated. For a window, the half-open range
`[source_sentence_start, source_sentence_end_exclusive)` says exactly which
source-document sentences it contains. For example, start `10` and end `15`
means original sentences 10, 11, 12, 13, and 14, in that order.

Inside `sentences_1000.csv`, `relative_sentence_index` starts again at zero for
each window, while `source_sentence_index` remains the position in the original
paper. `start_char` and `end_char` are half-open offsets relative to the
window's `text`: `text[start_char:end_char]` reproduces the sentence exactly.
The normalized window text retains the original normalized material between
sentences, normally a single space.

Gold entities use the same window-relative, half-open character convention, so
`window_text[start_char:end_char] == entity_text`. Source token offsets remain
available for tracing an entity back to the original SciREX token sequence.

Some windows from the same paper can overlap, with preprocessing limiting pair
overlap to 50%. Therefore, do not treat every window as an independent paper
and do not randomly split windows. Keep all rows with the same `doc_id` in the
same train/dev/test partition. The original `source_split` is already retained
for this purpose.

## Stage 1 batch pilot

The reproducible development pilot uses 20 document-disjoint SciREX development
windows and checkpoints every completed LLM run:

```powershell
python scripts/annotation/select_scirex_pilot.py
python scripts/annotation/run_scirex_batch.py --method few_shot_structured --max-sentences 10
python scripts/annotation/evaluate_scirex_batch.py
```

See `docs/annotation/STAGE1_SCIREX_PILOT.md` for the protocol, current results,
limitations, and the rule that prompt tuning must remain on the development
split.

The annotation UI now generates a visible, editable prompt from the active
dataset and label schema. SciREX uses the clearer public label `Dataset`, which
the evaluator maps to the release label `Material`. Batch runs reproduce this
behavior with `--prompt-mode suggested`.

The frozen held-out evaluation uses complete windows. Passing
`--max-sentences 0` means full-window processing; `--workers 4` runs independent
documents concurrently. The production runner atomically checkpoints after
every successful sentence, records manifest/prompt hashes and token usage,
enforces call/rate/cost limits, and refuses to mix configurations in one output
file. Start with the network-free preflight:

```powershell
python scripts/annotation/check_scirex_study_readiness.py
```

The complete gated commands and reporting rules are in
`docs/annotation/SCIREX_1000_RUNBOOK.md`.
