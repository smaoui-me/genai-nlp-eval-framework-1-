# Contributing

## Development setup

Docker is the canonical environment:

```bash
docker compose --profile test run --rm test
docker compose --profile tools build evaluation
docker compose up --build
```

For local Python 3.11 development:

```bash
python -m venv .venv
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Change requirements

- Add or update tests for behavioral changes.
- Keep prompts, configs, model identifiers, and dataset splits explicit.
- Do not report an experiment without retained predictions and scores.
- Keep prompt-development and held-out evaluation data separate by `doc_id`.
- Preserve immutable model output and reviewer history in annotation exports.
- Use scientific American English in report and study documentation.
- Do not add secrets, raw organizational data, generated result directories,
  virtual environments, or LaTeX auxiliary files.

## Pull requests

Describe the user-visible change, validation commands, affected datasets, and
any compatibility or privacy implications. A pull request should pass the full
network-free test suite and the Docker runtime build.
