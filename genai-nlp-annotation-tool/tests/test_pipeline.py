"""End-to-end test of run_annotation with a fake LLM (no network).

Checks that each estimator makes the right number of calls, produces
confidences with the right meaning, and that the per-model breakdown the
comparison page relies on is populated.
"""
import sys, json, math, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import utils.extraction_methods as em
from utils.llm_client import LLMResponse, TokenLogprob
from utils.model_registry import ModelChoice, Provider

FAILS = []
def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond: FAILS.append(name)

TEXT = "Customer reported a delayed delivery in Munich after shipment from DHL Hub."
LABELS = ["location", "organization"]

# The tokens of that one sentence, so the fake model can return valid indices.
from utils.tokenizer import split_sentences
sent = split_sentences(TEXT)[0]
tok_texts = [t.text for t in sent.tokens]
i_munich = tok_texts.index("Munich")
i_dhl = tok_texts.index("DHL")

CALLS = []

def make_fake(behaviour):
    """behaviour(call_index, model) -> list of entity dicts to return."""
    def fake_call_llm_full(prompt, choice=None, want_logprobs=False, **kwargs):
        model = choice.model if choice else "default"
        idx = len(CALLS)
        CALLS.append({"model": model, "want_logprobs": want_logprobs, "temperature": kwargs.get("temperature")})
        ents = behaviour(idx, model)
        payload = json.dumps({"entities": ents})
        logprobs = []
        if want_logprobs:
            # One token per character keeps the offset maths trivial and exact.
            logprobs = [TokenLogprob(token=ch, logprob=math.log(0.9)) for ch in payload]
        return LLMResponse(text=payload, model_id=f"fake:{model}", token_logprobs=logprobs)
    return fake_call_llm_full

_prov = Provider(key="fake", label="Fake", endpoint="http://x", api_key="k", models=("m1", "m2"))
M1 = ModelChoice(provider=_prov, model="m1")
M2 = ModelChoice(provider=_prov, model="m2")

MUNICH = {"text": "Munich", "type": "location", "start": i_munich, "end": i_munich}
DHL    = {"text": "DHL", "type": "organization", "start": i_dhl, "end": i_dhl}

print("\n=== A. estimator='none' — one call, no scores ===")
CALLS.clear()
em.call_llm_full = make_fake(lambda i, m: [MUNICH, DHL])
passes, want_lp = em.build_passes("none", M1)
r = em.run_annotation(TEXT, LABELS, "zero_shot_structured", max_sentences=1, passes=passes, want_logprobs=want_lp, estimator="none")
check("1 LLM call", len(CALLS) == 1, str(len(CALLS)))
check("2 entities", len(r["entities"]) == 2, str(len(r["entities"])))
check("no confidence", all(e["confidence"] is None for e in r["entities"]))
check("logprobs not requested", CALLS[0]["want_logprobs"] is False)

print("\n=== B. estimator='logprob' — one call, scored ===")
CALLS.clear()
em.call_llm_full = make_fake(lambda i, m: [MUNICH, DHL])
passes, want_lp = em.build_passes("logprob", M1)
r = em.run_annotation(TEXT, LABELS, "zero_shot_structured", max_sentences=1, passes=passes, want_logprobs=want_lp, estimator="logprob")
check("still 1 LLM call (no extra cost)", len(CALLS) == 1, str(len(CALLS)))
check("logprobs requested", CALLS[0]["want_logprobs"] is True)
check("logprobs reported available", r["logprobs_available"] is True)
confs = [e["confidence"] for e in r["entities"]]
check("every entity scored", all(c is not None for c in confs), str(confs))
check("score ~0.9 as constructed", all(abs(c - 0.9) < 1e-6 for c in confs), str(confs))
check("conf_source recorded", all(e["conf_source"] == "logprob" for e in r["entities"]))

print("\n=== C. estimator='self_consistency' — K calls, vote-based ===")
CALLS.clear()
# Munich in all 3 runs; DHL only in the first.
em.call_llm_full = make_fake(lambda i, m: [MUNICH, DHL] if i == 0 else [MUNICH])
passes, want_lp = em.build_passes("self_consistency", M1, n_samples=3, sample_temperature=0.7)
r = em.run_annotation(TEXT, LABELS, "zero_shot_structured", max_sentences=1, passes=passes, want_logprobs=want_lp, estimator="self_consistency")
check("3 LLM calls", len(CALLS) == 3, str(len(CALLS)))
check("n_llm_calls reported", r["n_llm_calls"] == 3, str(r["n_llm_calls"]))
check("sampling temperature applied", all(c["temperature"] == 0.7 for c in CALLS), str([c["temperature"] for c in CALLS]))
by = {e["text"]: e for e in r["entities"]}
check("Munich 3/3 -> 1.0", by["Munich"]["confidence"] == 1.0, str(by["Munich"]["confidence"]))
check("DHL 1/3 -> 0.333", abs(by["DHL"]["confidence"] - 1/3) < 1e-9, str(by["DHL"]["confidence"]))
check("borderline entity kept, not dropped", "DHL" in by)
check("least confident first", r["entities"][0]["text"] == "DHL")

print("\n=== D. sampling recovers an entity a single greedy run misses ===")
CALLS.clear()
# Greedy run 1 misses DHL; two later samples find it.
em.call_llm_full = make_fake(lambda i, m: [MUNICH] if i == 0 else [MUNICH, DHL])
passes, want_lp = em.build_passes("self_consistency", M1, n_samples=3)
r = em.run_annotation(TEXT, LABELS, "zero_shot_structured", max_sentences=1, passes=passes, want_logprobs=want_lp, estimator="self_consistency")
texts = {e["text"] for e in r["entities"]}
check("DHL surfaced even though run 1 missed it", "DHL" in texts, str(texts))
check("and it is flagged as uncertain", abs([e for e in r["entities"] if e["text"] == "DHL"][0]["confidence"] - 2/3) < 1e-9)

print("\n=== E. estimator='model_agreement' — one call per model ===")
CALLS.clear()
# m1 finds both, m2 only Munich.
em.call_llm_full = make_fake(lambda i, m: [MUNICH, DHL] if m == "m1" else [MUNICH])
passes, want_lp = em.build_passes("model_agreement", M1, compare_choices=[M1, M2])
r = em.run_annotation(TEXT, LABELS, "zero_shot_structured", max_sentences=1, passes=passes, want_logprobs=want_lp, estimator="model_agreement")
check("2 LLM calls, one per model", len(CALLS) == 2, str(len(CALLS)))
check("both models called", {c["model"] for c in CALLS} == {"m1", "m2"}, str([c["model"] for c in CALLS]))
by = {e["text"]: e for e in r["entities"]}
check("agreed span -> 1.0", by["Munich"]["confidence"] == 1.0)
check("disputed span -> 0.5", by["DHL"]["confidence"] == 0.5)
check("conf_source = model_agreement", by["DHL"]["conf_source"] == "model_agreement")
check("voters name the model", by["DHL"]["voters"] == ["fake:m1"], str(by["DHL"]["voters"]))
check("per_pass_entities filled for both", set(r["per_pass_entities"].keys()) == {"fake:m1", "fake:m2"}, str(list(r["per_pass_entities"])))
check("m1 pass has 2 entities", len(r["per_pass_entities"]["fake:m1"]) == 2)
check("m2 pass has 1 entity", len(r["per_pass_entities"]["fake:m2"]) == 1)

print("\n=== F. model_agreement needs >= 2 models ===")
try:
    em.build_passes("model_agreement", M1, compare_choices=[M1])
    check("raises with one model", False)
except ValueError:
    check("raises with one model", True)

print("\n=== G. logprob estimator degrades gracefully when unsupported ===")
CALLS.clear()
def no_logprob_call(prompt, choice=None, want_logprobs=False, **kwargs):
    CALLS.append({"model": choice.model if choice else "?", "want_logprobs": want_logprobs})
    return LLMResponse(text=json.dumps({"entities": [MUNICH]}), model_id="fake:m1", token_logprobs=[])
em.call_llm_full = no_logprob_call
passes, want_lp = em.build_passes("logprob", M1)
r = em.run_annotation(TEXT, LABELS, "zero_shot_structured", max_sentences=1, passes=passes, want_logprobs=want_lp, estimator="logprob")
check("run still succeeds", len(r["entities"]) == 1)
check("confidence is None, not a fake number", r["entities"][0]["confidence"] is None)
check("logprobs_available False so UI can warn", r["logprobs_available"] is False)

print("\n=== H. multi-sentence document ===")
CALLS.clear()
LONG = "Munich is nice. DHL Hub is in Bonn. Berlin is the capital."
em.call_llm_full = make_fake(lambda i, m: [])
passes, want_lp = em.build_passes("self_consistency", M1, n_samples=2)
r = em.run_annotation(LONG, LABELS, "zero_shot_structured", max_sentences=3, passes=passes, want_logprobs=want_lp, estimator="self_consistency")
check("calls = sentences x passes", len(CALLS) == r["n_sentences"] * 2, f"{len(CALLS)} calls, {r['n_sentences']} sentences")
check("n_llm_calls matches", r["n_llm_calls"] == len(CALLS))

print("\n" + "=" * 60)
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}"); sys.exit(1)
print("All pipeline checks passed.")
