# NLP Lab report — Team 38

Eight-page technical report for the NLP Lab (Summer Semester 2026), developed
with Carl Zeiss AG. The current version documents the reproducible annotation
framework, verified ticket-classification and Few-NERD baselines, and the
completed SciREX document-scale study.

## Build

With TeX Live, an Overleaf-compatible installation, or MiKTeX plus Perl:

```bash
latexmk -pdf main.tex
```

After a successful build, publish the PDF at the repository root with the
descriptive project filename:

```bash
mv -f main.pdf ../../NLP_Lab_Project_Report.pdf
```

On a minimal Windows MiKTeX installation without Perl, use the equivalent
sequence (this is the sequence used to verify the current PDF):

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
Move-Item -Force main.pdf ..\..\NLP_Lab_Project_Report.pdf
```

On Overleaf, upload this complete folder, choose `main.tex` as the main
document, and use pdfLaTeX. Download the compiled file and save it as
`NLP_Lab_Project_Report.pdf` at the repository root.

## Folder layout

| Path | Purpose |
|---|---|
| `main.tex` | Title block and the ordered list of compiled sections. |
| `tumpaper.sty` | TUM-inspired formatting, packages, and shared macros. |
| `sections/*.tex` | Report content, split by section for clean collaboration. |
| `references.bib` | BibTeX database. |
| `figures/` | Architecture, review-interface, and branding figures. |
| `../../NLP_Lab_Project_Report.pdf` | Published report at the repository root. |

## Current status

- Eight pages including figures and bibliography.
- Builds without LaTeX errors, undefined citations, or undefined references.
- Thirty-six bibliography records; SciREX uses the official ACL Anthology
  publication metadata.
- Quantitative claims distinguish completed experiments from the prepared but
  not completed 1,000-window SciREX configuration.
- Ticket data is correctly described as synthetic, monetary cost with zero
  configured rates as unknown, and long-text inference as sentence-wise full
  coverage rather than document-context modeling.

Limitations and future work are integrated into the discussion and conclusion
to keep the compiled report within the eight-page limit.

## Editing conventions

- Cite with `\citep{key}` or `\citet{key}`; never type citation numbers.
- Cross-reference with labels and `\Cref{...}`; never hard-code section or
  table numbers.
- Use `\code{...}` for file and field names, `\labelname{...}` for labels, and
  `\best{...}` only for the best value in a comparable table.
- Keep one sentence per source line where practical so Git diffs stay clear.
- Every reported number must be traceable to a retained result artifact. State
  sample size, split, metric, and whether the condition was used for prompt
  selection or held-out evaluation.
- Do not imply causality when multiple settings changed, and do not present a
  prepared benchmark as a completed experiment.

## Adding a source

1. Add a stable BibTeX record to `references.bib`, preferably copied from the
   publisher or ACL Anthology.
2. Verify that the source supports the surrounding claim.
3. Use a stable key such as `firstauthorYEARkeyword`.
