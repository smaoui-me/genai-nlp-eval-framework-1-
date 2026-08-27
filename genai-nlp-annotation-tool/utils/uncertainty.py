"""
How sure was the model about each entity it suggested?

## Why this exists

The first version of the review screen showed every suggested span and asked
the reviewer to confirm each one. That works on a three-sentence demo. On a
real 15-line ticket it can be forty spans, and a reviewer who has to click
through forty plausible-looking rows stops reading them properly. The review
workflow therefore needs to identify which rows deserve the most attention.

That is a well-studied idea. Uncertainty sampling, i.e. spending the human's
time where the model is least sure, is the oldest trick in active learning,
and the human-LLM annotation literature reaches the same conclusion for this
exact setting: reviewers should verify selectively rather than uniformly.

## The three estimators

None of these is obviously best, so the tool implements all three and lets
the user pick. Cost is in extra LLM calls per sentence.

| Estimator          | Cost | Needs                    | Measures |
|--------------------|------|--------------------------|----------|
| `logprob`          | 1x   | endpoint returns logprobs| how sure the model was about the tokens it wrote |
| `self_consistency` | Kx   | nothing special          | how often the span survives re-sampling |
| `model_agreement`  | Mx   | 2+ configured models      | whether different models agree |

`logprob` is the cheapest and should be the default when the endpoint
supports it. It has one real blind spot, though, and it is worth stating
because it decides when the extra cost of sampling is justified: a logprob
can only score spans the model *did* produce. It says nothing about the span
the model failed to produce. Sampling at a non-zero temperature does surface
those: an entity that appears in two runs out of five is a genuine borderline
candidate that a single greedy call would have dropped silently. So the two
methods measure different things. Logprobs grade precision; re-sampling also
recovers recall.

`model_agreement` is the same idea across models instead of across samples,
and it doubles as the model-comparison feature — the disagreements are both
the interesting rows for a reviewer and the interesting cases for deciding
whether a cheaper model is good enough.

## What the score means

Everything below produces `confidence` in 0..1, and `uncertainty` is simply
`1 - confidence`. The numbers are *not* calibrated probabilities: a 0.8 does
not mean "correct 80% of the time". They are a ranking signal, which is all
the review queue needs. Treating them as calibrated would be a mistake —
LLM confidence is generally overconfident.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

# Which estimators exist, and what to show in the dropdown.
ESTIMATORS = {
    "none": {
        "label": "None — single run, no scoring",
        "description": "One call per sentence. Fastest and cheapest, but every row has to be checked by hand.",
        "calls_per_sentence": 1,
    },
    "logprob": {
        "label": "Token confidence (1 call)",
        "description": (
            "Asks the endpoint for token log-probabilities and scores each entity by how sure "
            "the model was about the tokens it wrote. No extra cost. Falls back automatically "
            "if the endpoint does not return them."
        ),
        "calls_per_sentence": 1,
    },
    "self_consistency": {
        "label": "Self-consistency (K calls)",
        "description": (
            "Runs the same prompt K times at a non-zero temperature and scores each entity by "
            "how many runs found it. Costs K times as much, but also surfaces borderline "
            "entities a single run would miss entirely."
        ),
        "calls_per_sentence": None,  # depends on K
    },
    "model_agreement": {
        "label": "Model agreement (one call per model)",
        "description": (
            "Runs two or more models on the same text and scores each entity by how many of "
            "them found it. Doubles as a model comparison: the disagreements are where a "
            "cheaper model differs from the stronger one."
        ),
        "calls_per_sentence": None,  # depends on how many models
    },
}

# Anything below this is treated as high uncertainty when using a fixed cut-off.
#
# 0.80 is not a guess. We swept every possible cut-off against Few-NERD gold
# spans (scripts/annotation/run_uncertainty_experiment.py in the eval-framework
# repo). At 0.80, self-consistency flags 21% of spans, those spans are 3.8x
# more likely to be wrong than the rest, and they contain half of all the
# errors in the document. Lower cut-offs catch fewer errors; higher ones flag
# so much that the reviewer is back to reading everything.
HIGH_UNCERTAINTY_BELOW = 0.80
DEFAULT_REVIEW_THRESHOLD = HIGH_UNCERTAINTY_BELOW

# Share of spans to send for review when using the (default) budget mode.
DEFAULT_REVIEW_BUDGET = 0.20

# Why there are two modes, and why the budget one is the default:
#
# We measured both estimators against Few-NERD gold spans
# (scripts/annotation/run_uncertainty_experiment.py in the eval-framework
# repo). They separate right from wrong equally well — AUROC 0.68 for both —
# but their scores live on completely different scales. Self-consistency with
# K=5 can only return 0.0, 0.2, 0.4, 0.6, 0.8 or 1.0. Token log-probabilities
# pile up just under 1.0, with the lowest span in a 75-span run still scoring
# 0.789. A fixed 0.75 cut-off therefore flags a sensible 17% of the sampled
# spans and *nothing at all* of the log-probability ones, which makes the
# cheaper estimator look useless when it is not.
#
# Ranking by confidence and reviewing the lowest slice avoids that entirely:
# it works the same way whatever scale the estimator produces, and it lets a
# reviewer say "I have time for twenty spans" instead of guessing a number.
FLAG_MODES = {
    "budget": {
        "label": "Review the least confident share",
        "description": (
            "Ranks the spans and flags the bottom slice. Works on any estimator, since it only "
            "uses the ordering, and lets you set the review effort directly."
        ),
    },
    "threshold": {
        "label": "Fixed confidence cut-off",
        "description": (
            "Flags every span below a fixed number. Simple, but the right number differs per "
            "estimator: log-probabilities sit near 1.0, sampled votes are coarse steps."
        ),
    },
}


def flag_indices(entities: list[dict], mode: str = "budget", threshold: float = DEFAULT_REVIEW_THRESHOLD,
                 budget: float = DEFAULT_REVIEW_BUDGET) -> set[int]:
    """Which entities (by index) should be put in front of a human.

    Unscored entities are always flagged: no score is not evidence of being
    right, and hiding those rows would be the one failure mode we cannot
    detect afterwards.
    """
    unscored = {i for i, e in enumerate(entities) if e.get("confidence") is None}
    scored = [(i, e["confidence"]) for i, e in enumerate(entities) if e.get("confidence") is not None]

    if mode == "threshold":
        return unscored | {i for i, c in scored if c < threshold}

    # budget mode: take the lowest-confidence slice, but never split a group of
    # ties — if the cut lands in the middle of every span scoring 0.8, include
    # all of them, otherwise which ones get reviewed is arbitrary.
    if not scored:
        return unscored
    k = max(1, round(len(scored) * budget)) if budget > 0 else 0
    if k == 0:
        return unscored

    scored.sort(key=lambda pair: pair[1])
    cutoff_value = scored[min(k, len(scored)) - 1][1]
    top_value = scored[-1][1]

    # If the cut-off reaches the highest score, every span is tied and there is
    # no ranking to act on. Flag nothing rather than everything: "the model was
    # equally sure about all of them" is not a reason to re-read all of them,
    # and flagging 100% would make the feature worse than useless.
    if cutoff_value >= top_value:
        return unscored

    # Same idea for scores that are technically different but practically
    # identical (see MIN_USEFUL_SPREAD). Picking "the lowest 20%" out of
    # 0.99998 ... 1.00002 would flag rows for no reason a human could see.
    if confidence_spread(entities) < MIN_USEFUL_SPREAD:
        return unscored

    return unscored | {i for i, c in scored if c <= cutoff_value}


# The smallest gap between the most and least confident span that we are
# willing to call a ranking.
#
# Why this exists: on easy text, token confidences come back like 0.99998,
# 0.99999, 1.00002 — six "different" numbers spread over 0.00007. Counting
# distinct values says "we have a signal" and the review queue then flags
# three rows that all display as 1.00, which is baffling for the reviewer.
# Below this spread we treat the estimator as having told us nothing.
MIN_USEFUL_SPREAD = 0.01


def confidence_spread(entities: list[dict]) -> float:
    """Distance between the most and least confident scored span."""
    values = [e["confidence"] for e in entities if e.get("confidence") is not None]
    return (max(values) - min(values)) if len(values) > 1 else 0.0


def has_signal(entities: list[dict]) -> bool:
    """Did the estimator actually separate the spans from each other?

    False when every scored span got the same number, or numbers so close
    together that ordering them is meaningless. The UI uses this to say so
    plainly instead of showing a confidence column that reads 1.00 all the way
    down next to a review queue that looks arbitrary.
    """
    return confidence_spread(entities) >= MIN_USEFUL_SPREAD


def span_key(entity: dict) -> tuple:
    """What makes two suggested entities 'the same entity'.

    Character offsets plus the label. Deliberately strict: a span with the
    same text but different offsets is a different span, because for gold
    data the position is part of the answer.
    """
    return (entity["start"], entity["end"], entity["type"])


# ---------------------------------------------------------------------------
# 1. Voting-based estimators (self-consistency and model agreement share this)
# ---------------------------------------------------------------------------


@dataclass
class VoteResult:
    """One entity plus the record of who voted for it."""

    entity: dict
    votes: int
    total: int
    voters: list[str]  # run ids or model ids that produced this span

    @property
    def confidence(self) -> float:
        return self.votes / self.total if self.total else 0.0


def aggregate_votes(runs: list[tuple[str, list[dict]]]) -> list[VoteResult]:
    """Merge several runs into one entity list scored by how often each appeared.

    Args:
        runs: a list of (run_id, entities) pairs. `run_id` is a label for who
            produced the list — "run 1", or a model id for model agreement.

    Returns one VoteResult per distinct span, ordered by confidence (lowest
    first, so the rows that need attention come first) and then by position.
    """
    total = len(runs)
    if total == 0:
        return []

    by_key: dict[tuple, dict] = {}
    voters: dict[tuple, list[str]] = defaultdict(list)

    for run_id, entities in runs:
        # A single run can in principle emit the same span twice; count it once.
        seen_this_run = set()
        for entity in entities:
            key = span_key(entity)
            if key in seen_this_run:
                continue
            seen_this_run.add(key)
            by_key.setdefault(key, entity)
            voters[key].append(run_id)

    results = [
        VoteResult(entity=by_key[key], votes=len(v), total=total, voters=v)
        for key, v in voters.items()
    ]
    results.sort(key=lambda r: (r.confidence, r.entity["start"]))
    return results


# ---------------------------------------------------------------------------
# 2. Log-probability estimator
# ---------------------------------------------------------------------------


def confidence_from_logprobs(entity: dict, token_logprobs: list, raw_response: str) -> float | None:
    """Score one entity from the token logprobs of the response that produced it.

    The idea: find where this entity's text appears in the model's raw output,
    work out which generated tokens cover that stretch, and average their
    probabilities. A model that hesitated while writing "Frankfurt warehouse"
    gives those tokens a lower probability than one that wrote it confidently.

    Returns None when we cannot line the tokens up with the text, which is
    normal for some providers, and the caller then falls back to another
    estimator.
    """
    if not token_logprobs or not raw_response:
        return None

    needle = str(entity.get("text", "")).strip()
    if not needle:
        return None

    # Walk the generated tokens, rebuilding the response as we go, so we know
    # each token's character span inside the raw text.
    spans: list[tuple[int, int, float]] = []
    cursor = 0
    for tl in token_logprobs:
        tok = tl.token
        if not tok:
            continue
        spans.append((cursor, cursor + len(tok), tl.logprob))
        cursor += len(tok)

    # Where does the entity text sit in the response? Use the reconstructed
    # string rather than raw_response, because the two can differ in
    # whitespace and we need offsets consistent with `spans`.
    rebuilt = "".join(tl.token for tl in token_logprobs)
    idx = rebuilt.find(needle)
    if idx == -1:
        return None
    start, end = idx, idx + len(needle)

    covering = [lp for (s, e, lp) in spans if s < end and e > start]
    if not covering:
        return None

    # Mean log-probability, converted back to a probability. Mean rather than
    # product so that a long entity is not penalised just for being long.
    #
    # The clamp is not cosmetic. A log-probability should never be above 0, but
    # endpoints round, and we have seen +1e-5 come back for a token the model
    # was completely sure about. exp() of that is 1.00005, which is not a
    # probability and looks like a bug in the table. Clamping to [0, 1] keeps
    # the column honest.
    mean_logprob = sum(covering) / len(covering)
    return float(min(1.0, max(0.0, math.exp(mean_logprob))))


# ---------------------------------------------------------------------------
# 3. Turning a confidence into something a reviewer can act on
# ---------------------------------------------------------------------------


def band(confidence: float | None, threshold: float = DEFAULT_REVIEW_THRESHOLD) -> str:
    """Bucket a confidence into a short label for the review table."""
    if confidence is None:
        return "unknown"
    if confidence >= max(threshold, 0.999):
        return "high"
    if confidence >= threshold:
        return "medium"
    return "low"


def needs_review(confidence: float | None, threshold: float = DEFAULT_REVIEW_THRESHOLD) -> bool:
    """Should this row be put in front of a human?

    Unknown confidence counts as needing review. Not scoring a span is not
    the same as being sure about it, and defaulting the other way would hide
    rows precisely when the signal failed.
    """
    if confidence is None:
        return True
    return confidence < threshold


def summarise(
    entities: list[dict],
    threshold: float = DEFAULT_REVIEW_THRESHOLD,
    mode: str = "budget",
    budget: float = DEFAULT_REVIEW_BUDGET,
) -> dict:
    """Counts for the metric row above the review table."""
    scored = [e for e in entities if e.get("confidence") is not None]
    flagged_idx = flag_indices(entities, mode=mode, threshold=threshold, budget=budget)
    return {
        "n_entities": len(entities),
        "n_scored": len(scored),
        "n_flagged": len(flagged_idx),
        "flagged_idx": flagged_idx,
        "mean_confidence": (sum(e["confidence"] for e in scored) / len(scored)) if scored else None,
        "share_flagged": (len(flagged_idx) / len(entities)) if entities else 0.0,
    }


def agreement_report(runs: list[tuple[str, list[dict]]]) -> dict:
    """Compare runs pairwise. Used by the model comparison page.

    Reports, for each pair, how many spans both produced and the Jaccard
    overlap of their span sets. Jaccard rather than plain agreement because
    the two models may find different numbers of entities, and we care about
    the union, not just one model's list.
    """
    sets = {run_id: {span_key(e) for e in entities} for run_id, entities in runs}
    ids = list(sets.keys())

    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = sets[ids[i]], sets[ids[j]]
            union = a | b
            inter = a & b
            pairs.append(
                {
                    "a": ids[i],
                    "b": ids[j],
                    "n_a": len(a),
                    "n_b": len(b),
                    "shared": len(inter),
                    "only_a": len(a - b),
                    "only_b": len(b - a),
                    "jaccard": (len(inter) / len(union)) if union else 1.0,
                }
            )

    all_keys = set().union(*sets.values()) if sets else set()
    unanimous = [k for k in all_keys if all(k in s for s in sets.values())]
    return {
        "pairs": pairs,
        "n_union": len(all_keys),
        "n_unanimous": len(unanimous),
        "share_unanimous": (len(unanimous) / len(all_keys)) if all_keys else 0.0,
    }
