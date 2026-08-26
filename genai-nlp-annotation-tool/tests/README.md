# Tests

No pytest, no fixtures — plain scripts you run with python. Each prints
one line per check and exits non-zero if anything failed.

```bash
.venv/bin/python tests/test_uncertainty.py   # scoring maths, flagging, export
.venv/bin/python tests/test_pipeline.py      # run_annotation with a fake LLM
.venv/bin/python tests/test_app.py           # every Streamlit page renders
.venv/bin/python tests/test_review_ui.py     # split review and highlighting
.venv/bin/python tests/test_scirex_loader.py # processed fixture/source metadata
```

None of them makes a network call. `test_pipeline.py` swaps the LLM client for
a stub, so the vote counting and confidence maths are checked against known
answers rather than against whatever a model happens to say that day.

`test_app.py` uses Streamlit's own `AppTest`, which runs a page the way the
server would and lets us assert on the widgets without a browser.
