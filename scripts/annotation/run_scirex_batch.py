"""Production-oriented, resumable SciREX batch annotation runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "genai-nlp-annotation-tool"
sys.path.insert(0, str(APP_DIR))

from utils.extraction_methods import METHODS, extract_sentence  # noqa: E402
from utils.model_registry import available_choices, choice_by_id, default_choice  # noqa: E402
from utils.prompt_builder import prompt_hash, suggest_prompt  # noqa: E402
from utils.tokenizer import split_sentences  # noqa: E402

LABELS = ["Method", "Task", "Metric", "Dataset"]


class BudgetExceeded(RuntimeError):
    """The configured run safety budget has been reached."""


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    # OneDrive and antivirus scanners can briefly hold a newly written file on
    # Windows. Retrying the atomic replace preserves the previous checkpoint
    # until the new one can be installed instead of failing an otherwise valid
    # LLM run.
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.05 * (2 ** attempt), 1.0))


def checkpoint_path(directory: Path, example_id: str) -> Path:
    return directory / f"{hashlib.sha1(example_id.encode()).hexdigest()}.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def estimated_cost(input_tokens: int, output_tokens: int, input_rate: float, output_rate: float) -> float:
    return input_tokens * input_rate / 1_000_000 + output_tokens * output_rate / 1_000_000


class RunControl:
    """Thread-safe global rate limiter and run budget."""

    def __init__(
        self, *, calls: int = 0, input_tokens: int = 0, output_tokens: int = 0,
        usage_missing: int = 0, max_calls: int = 0, max_cost: float = 0.0,
        input_rate: float = 0.0, output_rate: float = 0.0, requests_per_minute: float = 0.0,
    ):
        self.calls = calls
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.usage_missing = usage_missing
        self.max_calls = max_calls
        self.max_cost = max_cost
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self.next_request_at = 0.0
        self.lock = threading.Lock()

    @property
    def cost(self) -> float:
        return estimated_cost(self.input_tokens, self.output_tokens, self.input_rate, self.output_rate)

    def before_call(self) -> None:
        with self.lock:
            if self.max_calls and self.calls >= self.max_calls:
                raise BudgetExceeded(f"Maximum logical call budget reached ({self.max_calls})")
            if self.max_cost and self.cost >= self.max_cost:
                raise BudgetExceeded(f"Maximum estimated spend reached (${self.max_cost:.2f})")
            now = time.monotonic()
            wait = max(0.0, self.next_request_at - now)
            self.next_request_at = max(now, self.next_request_at) + self.interval
            self.calls += 1  # reserve before sending so parallel workers cannot exceed max_calls
        if wait:
            time.sleep(wait)

    def after_call(self, input_tokens: int, output_tokens: int, usage_reported: bool) -> None:
        with self.lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            if not usage_reported:
                self.usage_missing += 1

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "logical_calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "usage_missing_calls": self.usage_missing,
                "estimated_cost_usd": round(self.cost, 6),
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="YAML runner defaults; explicit CLI flags win")
    parser.add_argument("--input", type=Path, default=Path("eval/corpora/scirex_dev_pilot_20.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/annotation/scirex_dev_pilot_20/predictions.jsonl"))
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--method", choices=METHODS, default="few_shot_structured")
    parser.add_argument("--prompt-mode", choices=("static", "suggested"), default="static")
    parser.add_argument("--model", help="Model choice id, e.g. hosted:deployment")
    parser.add_argument("--max-sentences", type=int, default=10, help="0 processes complete windows")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--requests-per-minute", type=float, default=0, help="0 disables rate limiting")
    parser.add_argument("--max-calls", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--max-cost-usd", type=float, default=0, help="0 means unlimited")
    parser.add_argument("--input-cost-per-million", type=float, default=0)
    parser.add_argument("--output-cost-per-million", type=float, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.config:
        values = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        values = values.get("runner", values)
        supplied = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
        actions = {action.dest: action for action in parser._actions}  # argparse has no public lookup
        for key, value in values.items():
            option = "--" + key.replace("_", "-")
            if key not in actions:
                raise ValueError(f"Unknown runner config key: {key}")
            if option in supplied:
                continue
            converter = actions[key].type
            converted = converter(value) if converter and value is not None else value
            if actions[key].choices and converted not in actions[key].choices:
                raise ValueError(f"Invalid value for {key}: {converted!r}")
            setattr(args, key, converted)
    return args


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.max_calls < 0 or args.max_cost_usd < 0:
        raise ValueError("workers must be >=1 and budgets cannot be negative")
    if args.max_cost_usd and not (args.input_cost_per_million or args.output_cost_per_million):
        raise ValueError("A spend budget requires input/output cost-per-million values")

    records = read_jsonl(args.input)
    if args.limit is not None:
        records = records[:args.limit]
    choice = choice_by_id(args.model) if args.model else default_choice()
    if choice is None:
        raise RuntimeError(f"No configured LLM model found. Available: {[x.id for x in available_choices()]}")

    sentence_limit = args.max_sentences if args.max_sentences > 0 else None
    sentences_by_id = {
        row["example_id"]: split_sentences(row["text"], max_sentences=sentence_limit) for row in records
    }
    exact_planned_calls = sum(len(value) for value in sentences_by_id.values())

    if args.prompt_mode == "suggested":
        prompt_template = suggest_prompt(LABELS, "SciREX", structured=METHODS[args.method]["structured"])
        prompt_sha256 = prompt_hash(prompt_template)
    else:
        prompt_template = None
        prompt_path = APP_DIR / "prompts" / f"{args.method}.txt"
        prompt_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()

    config = {
        "method": args.method, "model": choice.id, "labels": LABELS,
        "sentence_limit": sentence_limit, "temperature": 0.0,
        "prompt_mode": args.prompt_mode, "prompt_sha256": prompt_sha256,
        "manifest_sha256": file_sha256(args.input),
        "max_tokens": args.max_tokens,
    }
    print(
        f"Plan: {len(records)} examples, {exact_planned_calls} exact app calls, "
        f"method={args.method}, prompt={args.prompt_mode}, model={choice.id}", flush=True,
    )
    print(f"Manifest SHA-256: {config['manifest_sha256']}", flush=True)
    print(
        f"Safety: workers={args.workers}, rpm={args.requests_per_minute or 'unlimited'}, "
        f"max_calls={args.max_calls or 'unlimited'}, max_cost=${args.max_cost_usd or 0:.2f}", flush=True,
    )
    if args.dry_run:
        return

    existing = read_jsonl(args.output)
    for row in existing:
        found = {key: row.get(key, "static" if key == "prompt_mode" else None) for key in config}
        if found != config:
            raise RuntimeError(f"Refusing to mix run configurations in {args.output}: {found} != {config}")
    completed = {row["example_id"] for row in existing if row.get("status") == "ok"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or args.output.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_states = {}
    for example in records:
        if example["example_id"] in completed:
            continue
        path = checkpoint_path(checkpoint_dir, example["example_id"])
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("config") != config:
                raise RuntimeError(f"Checkpoint configuration mismatch: {path}")
            checkpoint_states[example["example_id"]] = state

    prior_rows = [row for row in existing if row.get("status") == "ok"] + list(checkpoint_states.values())
    control = RunControl(
        calls=sum(row.get("attempted_calls", row.get("llm_calls", 0)) for row in prior_rows),
        input_tokens=sum(row.get("input_tokens", 0) for row in prior_rows),
        output_tokens=sum(row.get("output_tokens", 0) for row in prior_rows),
        usage_missing=sum(row.get("usage_missing_calls", 0) for row in prior_rows),
        max_calls=args.max_calls, max_cost=args.max_cost_usd,
        input_rate=args.input_cost_per_million, output_rate=args.output_cost_per_million,
        requests_per_minute=args.requests_per_minute,
    )
    stop_event = threading.Event()
    run_started = time.monotonic()

    def run_one(index: int, example: dict) -> tuple[int, dict]:
        example_id = example["example_id"]
        sentences = sentences_by_id[example_id]
        path = checkpoint_path(checkpoint_dir, example_id)
        state = checkpoint_states.get(example_id) or {
            "example_id": example_id, "doc_id": example["doc_id"],
            "source_split": example["source_split"], "length_bucket": example["length_bucket"],
            **config, "config": config, "status": "running", "next_sentence": 0,
            "predictions": [], "llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
            "attempted_calls": 0,
            "total_tokens": 0, "usage_missing_calls": 0, "invalid_response_count": 0,
            "elapsed_seconds": 0.0, "started_at": datetime.now(timezone.utc).isoformat(),
        }
        started = time.monotonic()
        try:
            for sentence_index in range(int(state["next_sentence"]), len(sentences)):
                if stop_event.is_set():
                    raise InterruptedError("Run interrupted; sentence checkpoint preserved")
                control.before_call()
                state["attempted_calls"] += 1
                atomic_json(path, state)  # reserve durably before the external side effect
                result = extract_sentence(
                    args.method, LABELS, sentences[sentence_index],
                    llm_params={"temperature": 0.0, "max_tokens": args.max_tokens,
                                "timeout": args.timeout, "max_retries": args.max_retries},
                    choice=choice, prompt_template=prompt_template,
                )
                control.after_call(result.input_tokens, result.output_tokens, result.usage_reported)
                state["predictions"].extend(result.entities)
                state["llm_calls"] += 1
                state["input_tokens"] += result.input_tokens
                state["output_tokens"] += result.output_tokens
                state["total_tokens"] += result.total_tokens
                state["usage_missing_calls"] += int(not result.usage_reported)
                state["invalid_response_count"] += int(result.json_valid is False)
                state["next_sentence"] = sentence_index + 1
                state["processed_char_end"] = sentences[sentence_index].doc_start + len(sentences[sentence_index].text)
                state["elapsed_seconds"] = round(state["elapsed_seconds"] + time.monotonic() - started, 3)
                started = time.monotonic()
                atomic_json(path, state)  # sentence-level durable checkpoint
                if args.max_cost_usd and not result.usage_reported:
                    raise BudgetExceeded("Provider omitted token usage, so the spend budget cannot be enforced safely")
                if args.progress_every and state["next_sentence"] % args.progress_every == 0:
                    snapshot = control.snapshot()
                    elapsed = max(0.001, time.monotonic() - run_started)
                    rate = max(0.001, snapshot["logical_calls"] / elapsed)
                    remaining = max(0, exact_planned_calls - snapshot["logical_calls"])
                    print(
                        f"[{index}/{len(records)}] sentence {state['next_sentence']}/{len(sentences)}; "
                        f"global calls={snapshot['logical_calls']}; ETA~{remaining / rate / 3600:.2f}h; "
                        f"cost=${snapshot['estimated_cost_usd']:.4f}", flush=True,
                    )
            state.update({
                "status": "ok", "processed_sentences": len(sentences),
                "total_sentences": len(split_sentences(example["text"])),
                "processed_char_end": max(
                    (sentence.doc_start + len(sentence.text) for sentence in sentences), default=0
                ),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "estimated_cost_usd": round(estimated_cost(
                    state["input_tokens"], state["output_tokens"],
                    args.input_cost_per_million, args.output_cost_per_million,
                ), 6),
            })
            atomic_json(path, state)
            return index, {key: value for key, value in state.items() if key not in {"config", "next_sentence"}}
        except Exception as exc:
            state["status"] = "checkpointed"
            state["last_error_type"] = type(exc).__name__
            state["last_error"] = str(exc)
            atomic_json(path, state)
            return index, {
                "example_id": example_id, "doc_id": example["doc_id"], **config,
                "source_split": example["source_split"], "length_bucket": example["length_bucket"],
                "status": "error", "error_type": type(exc).__name__, "error": str(exc),
                "processed_sentences": state["next_sentence"], "llm_calls": state["llm_calls"],
                "attempted_calls": state["attempted_calls"],
                "input_tokens": state["input_tokens"], "output_tokens": state["output_tokens"],
                "total_tokens": state["total_tokens"], "usage_missing_calls": state["usage_missing_calls"],
                "elapsed_seconds": state["elapsed_seconds"],
            }

    pending = [(i, row) for i, row in enumerate(records, 1) if row["example_id"] not in completed]
    for index, example in enumerate(records, 1):
        if example["example_id"] in completed:
            print(f"[{index}/{len(records)}] skip completed {example['example_id']}", flush=True)

    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = [executor.submit(run_one, index, example) for index, example in pending]
    outputs_this_run = []
    was_interrupted = False
    try:
        for future in as_completed(futures):
            index, output = future.result()
            outputs_this_run.append(output)
            with args.output.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
            print(
                f"[{index}/{len(records)}] {output['status']} {output['example_id']} "
                f"sentences={output.get('processed_sentences', 0)} calls={output.get('llm_calls', 0)}",
                flush=True,
            )
            if output.get("error_type") == "BudgetExceeded":
                stop_event.set()
    except KeyboardInterrupt:
        print("Interrupt received; finishing active calls and preserving sentence checkpoints...", flush=True)
        stop_event.set()
        was_interrupted = True
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    snapshot = control.snapshot()
    print("Run totals: " + json.dumps(snapshot, sort_keys=True), flush=True)
    if args.max_cost_usd and snapshot["usage_missing_calls"]:
        raise RuntimeError(
            "Spend limit could not be fully verified because the provider omitted usage for "
            f"{snapshot['usage_missing_calls']} calls"
        )
    if was_interrupted:
        raise SystemExit(130)
    failures = [row for row in outputs_this_run if row.get("status") != "ok"]
    if failures:
        raise RuntimeError(
            f"Run incomplete: {len(failures)} examples checkpointed with errors. "
            "Fix the cause and rerun the identical command to resume."
        )


if __name__ == "__main__":
    main()
