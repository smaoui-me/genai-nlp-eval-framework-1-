# Runtime results

Evaluation scripts write predictions, checkpoints, debug records, token usage,
and score files under this directory. These files are intentionally ignored by
Git because they can be large and may contain submitted text or provider
metadata.

Every runner creates its required subdirectories automatically. Preserve study
outputs in controlled artifact storage and record the Git commit, config hash,
prompt hash, model identifier, and dataset manifest with the archived run.

Aggregated, non-sensitive results used in the scientific report are documented
under `docs/annotation/` and `docs/report/`.
