"""Tests for the split review view: span numbering, cross-references, and
the high-uncertainty-first behaviour. No network calls.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.annotation_store import (
    assign_span_numbers, entities_to_dataframe, render_highlighted_html,
)
from utils.uncertainty import (
    HIGH_UNCERTAINTY_BELOW, MIN_USEFUL_SPREAD, confidence_from_logprobs,
    confidence_spread, flag_indices, has_signal, summarise,
)

FAILS = []
def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond: FAILS.append(name)

TEXT = "Munich is nice and Berlin is big and Bonn is small and Hamburg is wet."
def ent(id_, txt, conf, status="pending"):
    s = TEXT.index(txt)
    return {"id": id_, "text": txt, "type": "location", "start": s, "end": s + len(txt),
            "source": "model", "model_name": "m", "status": status, "confidence": conf}

ENTS = [ent("a", "Munich", 1.0), ent("b", "Berlin", 0.4),
        ent("c", "Bonn", 0.6), ent("d", "Hamburg", None)]

print("\n=== 1. Span numbering follows reading order, not list order ===")
shuffled = [ENTS[3], ENTS[0], ENTS[2], ENTS[1]]
nums = assign_span_numbers(shuffled)
check("Munich is #1", nums["a"] == 1, str(nums))
check("Berlin is #2", nums["b"] == 2)
check("Bonn is #3", nums["c"] == 3)
check("Hamburg is #4", nums["d"] == 4)
check("deleted spans are not numbered",
      "x" not in assign_span_numbers(ENTS + [ent("x", "nice", 0.1, status="deleted")]))

print("\n=== 2. The '#' column matches the number in the text ===")
df = entities_to_dataframe(ENTS, numbers=nums)
check("'num' column exists", "num" in df.columns)
by_id = {r["id"]: r["num"] for _, r in df.iterrows()}
check("table numbers equal text numbers", by_id == nums, f"{by_id} vs {nums}")
check("num is an integer column", str(df["num"].dtype) == "int64", str(df["num"].dtype))
check("empty table still has num column", "num" in entities_to_dataframe([]).columns)

print("\n=== 3. Anchors and focus styling in the rendered text ===")
html = render_highlighted_html(TEXT, ENTS, ["location"], numbers=nums, focus_id="b",
                               flagged_ids={"b", "d"})
check("every span gets an id anchor", all(f'id="span-{i}"' in html for i in "abcd"))
check("focused span gets the thick outline", "3px solid" in html)
check("focused span gets a glow", "box-shadow" in html)
check("unfocused spans keep the normal border", "2px dashed" in html)
check("flagged spans get a warning mark", html.count("&#9888;") == 2, str(html.count("&#9888;")))
check("span numbers are printed in the text", ">1<" in html and ">2<" in html)
check("no focus id -> no glow", "box-shadow" not in render_highlighted_html(TEXT, ENTS, ["location"], numbers=nums))
check("text is html-escaped", "&lt;" in render_highlighted_html("a <b> c", [], ["location"]))

print("\n=== 4. What counts as high uncertainty ===")
check("the documented cut-off is 0.80", HIGH_UNCERTAINTY_BELOW == 0.80, str(HIGH_UNCERTAINTY_BELOW))
idx = flag_indices(ENTS, mode="threshold", threshold=HIGH_UNCERTAINTY_BELOW)
check("0.4 and 0.6 are flagged, 1.0 is not", idx == {1, 2, 3}, str(idx))
check("unscored span is flagged too", 3 in idx)
check("a span exactly at 0.80 is NOT flagged (strictly below)",
      flag_indices([{"confidence": 0.80}], mode="threshold", threshold=0.80) == set())
check("a span just under 0.80 is flagged",
      flag_indices([{"confidence": 0.7999}], mode="threshold", threshold=0.80) == {0})

print("\n=== 5. Splitting the table: flagged first, rest hidden ===")
stats = summarise(ENTS, mode="threshold", threshold=HIGH_UNCERTAINTY_BELOW)
flagged_ids = {ENTS[i]["id"] for i in stats["flagged_idx"]}
full = entities_to_dataframe(ENTS, numbers=nums)
flagged = full[full["id"].isin(flagged_ids)]
rest = full[~full["id"].isin(flagged_ids)]
check("3 rows in the 'needs a look' table", len(flagged) == 3, str(len(flagged)))
check("1 row hidden in the collapsed section", len(rest) == 1, str(len(rest)))
check("every row appears exactly once across both", len(flagged) + len(rest) == len(full))
check("no row is in both tables", set(flagged["id"]) & set(rest["id"]) == set())
check("the confident one is the hidden one", list(rest["id"]) == ["a"], str(list(rest["id"])))

print("\n=== 6. Edge cases ===")
check("no entities at all", summarise([])["n_flagged"] == 0)
allsame = [ent("a", "Munich", 1.0), ent("b", "Berlin", 1.0)]
s = summarise(allsame, mode="budget", budget=0.2)
check("all equally confident -> nothing flagged", s["n_flagged"] == 0, str(s["n_flagged"]))
noconf = [{"id": "a", "text": "Munich", "type": "location", "start": 0, "end": 6,
           "source": "model", "model_name": None, "status": "pending", "confidence": None}]
check("no confidence at all -> everything flagged", summarise(noconf)["n_flagged"] == 1)
check("...so nothing gets silently hidden",
      len(entities_to_dataframe(noconf, numbers=assign_span_numbers(noconf))) == 1)
onlydeleted = [ent("z", "Munich", 0.1, status="deleted")]
check("only deleted spans -> empty table", len(entities_to_dataframe(onlydeleted)) == 0)
check("only deleted spans -> render does not crash",
      isinstance(render_highlighted_html(TEXT, onlydeleted, ["location"]), str))
check("focus id that no longer exists is ignored",
      "box-shadow" not in render_highlighted_html(TEXT, ENTS, ["location"], numbers=nums, focus_id="gone"))
overlap = [ent("a", "Munich", 0.5),
           {"id": "b", "text": "Munich is", "type": "location", "start": 0, "end": 9,
            "source": "model", "model_name": None, "status": "pending", "confidence": 0.5}]
check("overlapping spans do not corrupt the html",
      render_highlighted_html(TEXT, overlap, ["location"]).count("<mark") == 1)

print("\n=== 7. Near-identical scores must not create a fake review queue ===")
# Real values observed from the hosted endpoint on easy Wikipedia text.
near = [{"confidence": c} for c in
        [0.9999771, 0.9999847, 0.9999847, 0.9999886, 1.0000153, 1.0000229, 1.0000229, 1.0000496]]
check("spread is far below the useful minimum", confidence_spread(near) < MIN_USEFUL_SPREAD,
      f"{confidence_spread(near):.7f} < {MIN_USEFUL_SPREAD}")
check("has_signal says no", has_signal(near) is False)
check("budget mode flags nothing", flag_indices(near, mode="budget", budget=0.2) == set(),
      str(flag_indices(near, mode="budget", budget=0.2)))
check("summarise agrees", summarise(near, mode="budget", budget=0.2)["n_flagged"] == 0)
spread_ok = [{"confidence": c} for c in [0.80, 0.95, 1.0, 1.0]]
check("a real 0.20 spread still ranks", has_signal(spread_ok) is True)
check("...and still flags the lowest", flag_indices(spread_ok, mode="budget", budget=0.25) == {0})

print("\n=== 8. Confidence can never leave [0, 1] ===")
class TL:
    def __init__(s, token, logprob): s.token, s.logprob = token, logprob
# A positive log-probability is not mathematically possible but endpoints
# round, and exp(+1e-5) is 1.00001. It must be clamped.
toks = [TL("Mun", 1e-5), TL("ich", 2e-5)]
c = confidence_from_logprobs({"text": "Munich"}, toks, "Munich")
check("positive logprob clamps to 1.0", c == 1.0, f"got {c}")
check("never exceeds 1", c <= 1.0)
neg = [TL("Mun", -5.0), TL("ich", -6.0)]
c2 = confidence_from_logprobs({"text": "Munich"}, neg, "Munich")
check("very negative logprob stays in range", 0.0 <= c2 <= 1.0, f"got {c2:.6f}")

print("\n=== 9. The jump link column ===")
df = entities_to_dataframe(ENTS, numbers=nums)
check("jump column exists", "jump" in df.columns)
check("jump points at the span anchor",
      all(r["jump"] == f"#span-{r['id']}" for _, r in df.iterrows()),
      str(df["jump"].tolist()[:2]))
html = render_highlighted_html(TEXT, ENTS, ["location"], numbers=nums)
check("every jump target exists in the html",
      all(f'id="span-{i}"' in html for i in df["id"]), "anchors present")
check("empty table still has jump column", "jump" in entities_to_dataframe([]).columns)

print("\n=== 10. Source column shows the model, not the provider prefix ===")
withprov = [dict(ENTS[0], model_name="hosted:gpt-5.4")]
d = entities_to_dataframe(withprov, numbers=assign_span_numbers(withprov))
check("prefix stripped", d.iloc[0]["source"] == "gpt-5.4", str(d.iloc[0]["source"]))
human = [dict(ENTS[0], source="human", model_name=None)]
check("human rows unaffected",
      entities_to_dataframe(human, numbers=assign_span_numbers(human)).iloc[0]["source"] == "human")

print("\n" + "=" * 60)
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}"); sys.exit(1)
print("All review-UI checks passed.")
