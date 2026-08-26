"""Offline tests for the new uncertainty / multi-model code.

No network: the LLM call is monkey-patched so we can check the aggregation
logic deterministically.
"""
import sys, types, math, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.uncertainty import (
    aggregate_votes, confidence_from_logprobs, span_key, band, needs_review,
    summarise, agreement_report, DEFAULT_REVIEW_THRESHOLD,
)
from utils.annotation_store import (
    entities_to_dataframe, compute_confidence_diagnostics, build_gold_export,
    compute_annotation_metrics,
)

FAILS = []
def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond:
        FAILS.append(name)

def ent(text, t, s, e):
    return {"text": text, "type": t, "start": s, "end": e}

print("\n=== 1. Vote aggregation (self-consistency) ===")
runs = [
    ("run 1", [ent("Munich", "location", 40, 46), ent("Berlin", "location", 155, 161)]),
    ("run 2", [ent("Munich", "location", 40, 46)]),
    ("run 3", [ent("Munich", "location", 40, 46), ent("Berlin", "location", 155, 161)]),
]
votes = aggregate_votes(runs)
by_text = {v.entity["text"]: v for v in votes}
check("Munich found in 3/3 -> confidence 1.0", by_text["Munich"].confidence == 1.0, f"got {by_text['Munich'].confidence}")
check("Berlin found in 2/3 -> confidence 0.667", abs(by_text["Berlin"].confidence - 2/3) < 1e-9, f"got {by_text['Berlin'].confidence:.3f}")
check("least confident sorts first", votes[0].entity["text"] == "Berlin")
check("voters recorded", by_text["Berlin"].voters == ["run 1", "run 3"], str(by_text["Berlin"].voters))

print("\n=== 2. Duplicate span inside one run counted once ===")
dup = [("run 1", [ent("Munich", "location", 40, 46), ent("Munich", "location", 40, 46)]),
       ("run 2", [])]
v = aggregate_votes(dup)
check("one entry for the duplicated span", len(v) == 1, f"got {len(v)}")
check("confidence 0.5 not 1.0", v[0].confidence == 0.5, f"got {v[0].confidence}")

print("\n=== 3. Same text, different offsets = different spans ===")
diff = [("run 1", [ent("Berlin", "location", 10, 16)]),
        ("run 2", [ent("Berlin", "location", 99, 105)])]
v = aggregate_votes(diff)
check("kept as two distinct spans", len(v) == 2, f"got {len(v)}")
check("each at 0.5", all(x.confidence == 0.5 for x in v))

print("\n=== 4. Logprob-based confidence ===")
class TL:
    def __init__(self, token, logprob): self.token, self.logprob = token, logprob
# model wrote: {"entities":[{"text":"Munich",...
toks = [TL('{"text": "', -0.01), TL("Mun", math.log(0.9)), TL("ich", math.log(0.8)), TL('"}', -0.01)]
raw = "".join(t.token for t in toks)
c = confidence_from_logprobs(ent("Munich", "location", 0, 6), toks, raw)
expected = math.exp((math.log(0.9) + math.log(0.8)) / 2)
check("mean logprob -> probability", c is not None and abs(c - expected) < 1e-6, f"got {c}, expected {expected:.4f}")
check("confidence in 0..1", 0 <= c <= 1)

c_missing = confidence_from_logprobs(ent("Paris", "location", 0, 5), toks, raw)
check("entity not in output -> None", c_missing is None)
check("no logprobs -> None", confidence_from_logprobs(ent("Munich","location",0,6), [], raw) is None)

print("\n=== 5. Long entity not penalised for length ===")
short = [TL("Bonn", math.log(0.5))]
long_ = [TL("Frank", math.log(0.5)), TL("furt", math.log(0.5)), TL(" ware", math.log(0.5)), TL("house", math.log(0.5))]
cs = confidence_from_logprobs(ent("Bonn","location",0,4), short, "Bonn")
cl = confidence_from_logprobs(ent("Frankfurt warehouse","location",0,19), long_, "Frankfurt warehouse")
check("equal per-token prob -> equal score regardless of length", abs(cs - cl) < 1e-9, f"{cs:.4f} vs {cl:.4f}")

print("\n=== 6. Review flagging ===")
check("None counts as needing review", needs_review(None) is True)
check("below threshold flagged", needs_review(0.4, 0.75) is True)
check("above threshold not flagged", needs_review(0.9, 0.75) is False)
check("band low", band(0.2, 0.75) == "low")
check("band high", band(1.0, 0.75) == "high")

ents = [
    {"text":"a","type":"location","start":0,"end":1,"status":"pending","confidence":0.3},
    {"text":"b","type":"location","start":2,"end":3,"status":"pending","confidence":0.95},
    {"text":"c","type":"location","start":4,"end":5,"status":"pending","confidence":None},
]
s = summarise(ents, 0.75)
check("2 of 3 flagged (low + unscored)", s["n_flagged"] == 2, str(s))
check("scored count excludes None", s["n_scored"] == 2)

print("\n=== 7. DataFrame: column + sorting ===")
rows = [
    {"id":"1","text":"high","type":"location","start":0,"end":4,"source":"model","model_name":"m","status":"pending","confidence":0.99},
    {"id":"2","text":"low","type":"location","start":10,"end":13,"source":"model","model_name":"m","status":"pending","confidence":0.20},
    {"id":"3","text":"none","type":"location","start":20,"end":24,"source":"model","model_name":"m","status":"pending","confidence":None},
]
df = entities_to_dataframe(rows)
check("confidence column present", "confidence" in df.columns)
check("confidence dtype is float", str(df["confidence"].dtype) == "float64", str(df["confidence"].dtype))
check("unscored row sorts first", df.iloc[0]["text"] == "none", df["text"].tolist().__str__())
check("then lowest confidence", df.iloc[1]["text"] == "low", df["text"].tolist().__str__())
check("highest last", df.iloc[2]["text"] == "high")
df_uns = entities_to_dataframe(rows, sort_by_uncertainty=False)
check("sorting can be turned off", df_uns.iloc[0]["text"] == "high")
check("empty frame has confidence col", "confidence" in entities_to_dataframe([]).columns)

print("\n=== 8. Does the flag predict human edits? (diagnostics) ===")
spans = [
    {"source":"model","status":"edited","confidence":0.2},
    {"source":"model","status":"deleted","confidence":0.3},
    {"source":"model","status":"confirmed","confidence":0.4},
    {"source":"model","status":"confirmed","confidence":0.9},
    {"source":"model","status":"confirmed","confidence":0.95},
    {"source":"model","status":"confirmed","confidence":0.99},
]
d = compute_confidence_diagnostics(spans, threshold=0.75)
check("3 flagged", d["n_flagged"] == 3, str(d["n_flagged"]))
check("flag precision 2/3", abs(d["flag_precision"] - 2/3) < 1e-3, str(d["flag_precision"]))  # value is rounded to 3dp
check("flag recall 2/2 = 1.0", d["flag_recall"] == 1.0, str(d["flag_recall"]))
check("lift is infinite (no unflagged changes)", d["lift"] == float("inf"), str(d["lift"]))
check("no scores -> flag unavailable", compute_confidence_diagnostics([{"source":"model","status":"confirmed"}])["confidence_available"] is False)

print("\n=== 9. Export carries confidence into the audit trail ===")
export_entities = [
    {"id":"1","text":"Munich","type":"location","start":0,"end":6,"source":"model","model_name":"project:gpt-5.4",
     "status":"confirmed","confidence":0.9,"conf_source":"logprob","voters":["run 1"]},
    {"id":"2","text":"Bad","type":"location","start":10,"end":13,"source":"model","model_name":"project:gpt-5.4",
     "status":"deleted","confidence":0.2,"conf_source":"logprob","voters":["run 1"]},
]
exp = build_gold_export("doc", "Munich xxx Bad", ["location"], "few_shot_structured", export_entities,
                        run_meta={"estimator":"logprob","n_llm_calls":3})
check("uncertainty meta stored", exp["uncertainty"]["estimator"] == "logprob")
check("review log keeps confidence", exp["review_log"][0]["confidence"] == 0.9)
check("review log keeps conf_source", exp["review_log"][1]["conf_source"] == "logprob")
check("deleted span excluded from gold", len(exp["gold_entities"]) == 1)
check("export is JSON-serialisable", json.dumps(exp) is not None)
m = compute_annotation_metrics(exp)
check("metrics include diagnostics", m["confidence_available"] is True, str(m.get("confidence_available")))

print("\n=== 10. Agreement report (model comparison) ===")
rep = agreement_report([
    ("modelA", [ent("Munich","location",40,46), ent("Berlin","location",155,161)]),
    ("modelB", [ent("Munich","location",40,46), ent("DHL","organization",67,70)]),
])
check("union of 3 distinct spans", rep["n_union"] == 3, str(rep["n_union"]))
check("1 unanimous", rep["n_unanimous"] == 1, str(rep["n_unanimous"]))
check("jaccard 1/3", abs(rep["pairs"][0]["jaccard"] - 1/3) < 1e-9, str(rep["pairs"][0]["jaccard"]))
check("only_a = 1", rep["pairs"][0]["only_a"] == 1)
check("only_b = 1", rep["pairs"][0]["only_b"] == 1)
identical = agreement_report([("a",[ent("X","location",0,1)]), ("b",[ent("X","location",0,1)])])
check("identical models -> jaccard 1.0", identical["pairs"][0]["jaccard"] == 1.0)
check("empty runs -> jaccard 1.0 not crash", agreement_report([("a",[]),("b",[])])["pairs"][0]["jaccard"] == 1.0)

print("\n" + "=" * 60)
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("All checks passed.")
