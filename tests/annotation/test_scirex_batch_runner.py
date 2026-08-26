import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_runner():
    path = ROOT / "scripts" / "annotation" / "run_scirex_batch.py"
    spec = importlib.util.spec_from_file_location("run_scirex_batch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sentence_checkpoint_resumes_without_repeating_success(monkeypatch, tmp_path):
    runner = load_runner()
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "predictions.jsonl"
    checkpoints = tmp_path / "checkpoints"
    example = {
        "example_id": "paper-1-window-1",
        "doc_id": "paper-1",
        "source_split": "dev",
        "length_bucket": "short",
        "text": "One sentence. Two sentence. Three sentence.",
        "entities": [],
    }
    manifest.write_text(json.dumps(example) + "\n", encoding="utf-8")
    monkeypatch.setattr(runner, "default_choice", lambda: SimpleNamespace(id="fake:model"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scirex_batch.py", "--input", str(manifest), "--output", str(output),
            "--checkpoint-dir", str(checkpoints), "--max-sentences", "0",
            "--workers", "1", "--requests-per-minute", "0", "--progress-every", "0",
        ],
    )

    first_attempts = 0

    def interrupted_extract(*args, **kwargs):
        nonlocal first_attempts
        first_attempts += 1
        if first_attempts == 2:
            raise RuntimeError("simulated interruption")
        return SimpleNamespace(
            entities=[], input_tokens=10, output_tokens=2, total_tokens=12,
            usage_reported=True, json_valid=True,
        )

    monkeypatch.setattr(runner, "extract_sentence", interrupted_extract)
    with pytest.raises(RuntimeError, match="Run incomplete"):
        runner.main()

    checkpoint_file = runner.checkpoint_path(checkpoints, example["example_id"])
    interrupted = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    assert interrupted["next_sentence"] == 1
    assert interrupted["llm_calls"] == 1
    assert interrupted["attempted_calls"] == 2

    resumed_calls = 0

    def successful_extract(*args, **kwargs):
        nonlocal resumed_calls
        resumed_calls += 1
        return SimpleNamespace(
            entities=[], input_tokens=10, output_tokens=2, total_tokens=12,
            usage_reported=True, json_valid=True,
        )

    monkeypatch.setattr(runner, "extract_sentence", successful_extract)
    runner.main()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["status"] == "error"
    assert rows[-1]["status"] == "ok"
    assert rows[-1]["processed_sentences"] == 3
    assert rows[-1]["llm_calls"] == 3
    assert rows[-1]["attempted_calls"] == 4
    assert rows[-1]["total_tokens"] == 36
    assert resumed_calls == 2


def test_run_control_enforces_reserved_call_budget():
    runner = load_runner()
    control = runner.RunControl(max_calls=2)
    control.before_call()
    control.before_call()
    try:
        control.before_call()
    except runner.BudgetExceeded:
        pass
    else:
        raise AssertionError("RunControl allowed a call above its budget")
    assert control.snapshot()["logical_calls"] == 2


def test_atomic_checkpoint_retries_transient_windows_lock(monkeypatch, tmp_path):
    runner = load_runner()
    destination = tmp_path / "checkpoint.json"
    original_replace = Path.replace
    attempts = 0

    def flaky_replace(self, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("simulated OneDrive lock")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    runner.atomic_json(destination, {"next_sentence": 7})

    assert attempts == 3
    assert json.loads(destination.read_text(encoding="utf-8")) == {"next_sentence": 7}
