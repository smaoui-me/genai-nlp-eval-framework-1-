"""
Helper functions for the Annotate page: keeping track of entities in
st.session_state, building the editable review table, drawing the
highlighted-text preview, and saving/loading gold JSON files.

Streamlit tip: st.session_state is a dict-like object that Streamlit keeps
between reruns (normal variables would reset every time). We store the
current document's entities there so they survive button clicks.

One "entity" (one highlighted span) is a plain dict shaped like this::

    {
        "id": "e3",                # unique ID we generate, used to find/update this entity later
        "text": "Munich",          # the text of the span
        "type": "location",        # the label assigned to it
        "start": 42,               # start position in the document (character count, inclusive)
        "end": 48,                 # end position (exclusive)
        "source": "model" | "human",       # who created it
        "model_name": "gpt-5.4" | None,    # which model, only set when source == "model"
        "status": "pending" | "confirmed" | "edited" | "deleted",
    }
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.labels import color_for_label

GOLD_EXPORT_DIR = Path(__file__).resolve().parent.parent / "data" / "gold_exports"


def init_session_state() -> None:
    """Create every session_state key this app needs, with a starting value.
    Safe to call more than once — setdefault only sets a value the first time."""
    st.session_state.setdefault("labels", None)
    st.session_state.setdefault("available_labels", None)
    st.session_state.setdefault("doc_name", None)
    st.session_state.setdefault("doc_text", "")
    st.session_state.setdefault("entities", [])
    st.session_state.setdefault("method_id", "few_shot_structured")
    st.session_state.setdefault("last_run_meta", None)
    st.session_state.setdefault("routing_result", None)
    st.session_state.setdefault("routing_approved_department", None)
    st.session_state.setdefault("exported_docs", [])
    # Bumped every time entities change from a button click (not from typing
    # in the table itself). We use it inside the data_editor's key so the
    # table always redraws with the latest data — see pages/1_annotate.py.
    st.session_state.setdefault("entities_version", 0)
    # Which span the reviewer last clicked "jump to" on, so the text view
    # can outline it and scroll to it.
    st.session_state.setdefault("focus_span_id", None)


def bump_entities_version() -> None:
    """Call this after changing st.session_state.entities from a button, so
    the review table below is forced to redraw with the new data."""
    st.session_state.entities_version += 1


def new_entity_id() -> str:
    """A short random ID, unique enough to tell entities apart in one document."""
    return uuid.uuid4().hex[:8]


def entities_to_dataframe(
    entities: list[dict],
    sort_by_uncertainty: bool = True,
    numbers: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Turn our list of entity dicts into a table (DataFrame) for st.data_editor.

    When `sort_by_uncertainty` is on, the least confident rows come first, so
    a reviewer working top-down spends their attention where the model was
    least sure. Rows with no confidence score sort first too, because "not
    scored" is not the same as "certain".
    """
    visible = [e for e in entities if e["status"] != "deleted"]
    if not visible:
        # An empty pd.DataFrame(columns=[...]) gives every column "object"
        # dtype, but the "confirmed"/"delete" checkbox columns and
        # "start"/"end" number columns below need real bool/int dtypes to
        # match st.data_editor's column_config — a mismatch here can crash
        # the app when it's serialized for display, so we set them by hand.
        return pd.DataFrame({
            "id": pd.Series(dtype="object"),
            "jump": pd.Series(dtype="object"),
            "num": pd.Series(dtype="int64"),
            "text": pd.Series(dtype="object"),
            "type": pd.Series(dtype="object"),
            "confidence": pd.Series(dtype="float64"),
            "start": pd.Series(dtype="int64"),
            "end": pd.Series(dtype="int64"),
            "source": pd.Series(dtype="object"),
            "status": pd.Series(dtype="object"),
            "confirmed": pd.Series(dtype="bool"),
            "delete": pd.Series(dtype="bool"),
        })

    df = pd.DataFrame(visible)
    # "#" ties each row to the same number printed on the highlight in the text.
    numbers = numbers or {}
    df["num"] = df["id"].map(numbers).fillna(0).astype("int64")
    # A same-page link to the anchor render_highlighted_html() puts on each
    # highlight. Streamlit's LinkColumn shows the number and makes it
    # clickable, which is why there is no separate "jump" button row.
    df["jump"] = "#span-" + df["id"].astype(str)
    if "confidence" not in df.columns:
        df["confidence"] = None
    # float64 with NaN for "not scored" — Streamlit's ProgressColumn needs a
    # real numeric dtype, and an object column of None/float would break it.
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").astype("float64")

    # A friendly "who suggested this" column: the model name if the LLM
    # suggested it, otherwise "human". model_name is stored as a full id like
    # "hosted:gpt-5.4" so runs stay traceable, but the provider prefix is noise
    # in a table of forty rows, so we show only the part after the colon.
    def _who(row):
        if row["source"] == "model" and row.get("model_name"):
            return str(row["model_name"]).split(":", 1)[-1]
        return row["source"]

    df["source"] = df.apply(_who, axis=1)

    # A checkbox column showing/controlling whether each span has been
    # reviewed. "edited" also counts as reviewed (you can't relabel
    # something without looking at it).
    df["confirmed"] = df["status"].isin(["confirmed", "edited"])
    df["delete"] = False

    if sort_by_uncertainty:
        # NaN (unscored) sorts first, then ascending confidence, then position.
        df = df.sort_values(
            by=["confidence", "start"], ascending=[True, True], na_position="first", kind="stable"
        ).reset_index(drop=True)

    return df[["id", "jump", "num", "text", "type", "confidence", "start", "end",
               "source", "status", "confirmed", "delete"]]


def apply_edits(entities: list[dict], edited_df: pd.DataFrame, labels: list[str]) -> list[dict]:
    """Copy changes from the (possibly edited) table back into our entity list."""
    by_id = {e["id"]: e for e in entities}  # quick lookup by id

    for _, row in edited_df.iterrows():
        entity = by_id.get(row["id"])
        if entity is None:
            continue

        if bool(row["delete"]):
            entity["status"] = "deleted"
            continue

        relabeled = entity["type"] != row["type"]
        entity["type"] = row["type"]

        if relabeled:
            entity["status"] = "edited"
        elif bool(row["confirmed"]):
            entity["status"] = "confirmed"
        else:
            entity["status"] = "pending"

    return list(by_id.values())


def confirm_all_pending(entities: list[dict]) -> list[dict]:
    """Mark every still-pending entity as confirmed (the "Confirm all" button)."""
    for e in entities:
        if e["status"] == "pending":
            e["status"] = "confirmed"
    return entities


def add_manual_entity(entities: list[dict], text: str, entity_type: str, doc_text: str) -> tuple[list[dict], str | None]:
    """Add an entity a human typed in by hand. Returns (updated_entities, error_message)."""
    start = doc_text.find(text)
    if start == -1:
        return entities, f"Could not find the exact text {text!r} in the document (check spelling/casing)."

    end = start + len(text)
    entities.append(
        {
            "id": new_entity_id(),
            "text": text,
            "type": entity_type,
            "start": start,
            "end": end,
            "source": "human",
            "model_name": None,
            "status": "confirmed",  # a human typed this on purpose, so it's already reviewed
        }
    )
    return entities, None


def assign_span_numbers(entities: list[dict]) -> dict[str, int]:
    """Number the visible spans 1, 2, 3... in the order they appear in the text.

    The same number is printed next to the highlight and in the review table's
    "#" column, so a reviewer can match a row to a place in the document at a
    glance, without clicking anything.
    """
    visible = sorted((e for e in entities if e["status"] != "deleted"), key=lambda e: e["start"])
    return {e["id"]: i + 1 for i, e in enumerate(visible)}


def render_highlighted_html(
    text: str,
    entities: list[dict],
    labels: list[str],
    numbers: dict[str, int] | None = None,
    focus_id: str | None = None,
    flagged_ids: set[str] | None = None,
) -> str:
    """Build an HTML snippet showing `text` with each entity highlighted.

    Args:
        numbers: id -> the small number printed on the highlight, from
            assign_span_numbers().
        focus_id: the span the reviewer selected in the table. It gets a thicker
            outline and an anchor the page scrolls to.
        flagged_ids: spans the confidence score marked as needing review. They
            get a warning dot so they stand out in the text as well as in the
            table.
    """
    numbers = numbers or {}
    flagged_ids = flagged_ids or set()
    visible = sorted((e for e in entities if e["status"] != "deleted"), key=lambda e: e["start"])

    pieces = []
    cursor = 0  # how far through `text` we've already added
    for e in visible:
        start, end = e["start"], e["end"]
        if start < cursor:
            continue  # overlapping span — skip it to keep the HTML valid

        pieces.append(_escape(text[cursor:start]))  # plain text before this entity

        color = color_for_label(e["type"], labels)
        border = "2px dashed" if e["status"] == "pending" else "1px solid"  # dashed = still needs review
        is_focus = e["id"] == focus_id
        if is_focus:
            # The selected row: strong outline plus a soft glow, so the eye
            # lands on it after the page scrolls here.
            border = "3px solid"
            extra = f"box-shadow:0 0 0 4px {color}55; scroll-margin-top:20px;"
        else:
            extra = ""

        num = numbers.get(e["id"])
        num_html = (
            f'<span style="font-size:0.62em;font-weight:700;color:{_darken(color)};'
            f'vertical-align:super;margin-right:2px;">{num}</span>' if num else ""
        )
        warn = (
            '<span title="flagged for review" style="font-size:0.7em;margin-left:3px;">&#9888;</span>'
            if e["id"] in flagged_ids else ""
        )

        pieces.append(
            f'<mark id="span-{_escape(str(e["id"]))}" '
            f'style="background:{color}33;border:{border} {color};border-radius:5px;'
            f'padding:1px 4px;margin:0 1px;{extra}">{num_html}{_escape(text[start:end])}'
            f'<span style="font-size:0.68em;font-weight:700;color:{_darken(color)};'
            f'text-transform:uppercase;margin-left:5px;letter-spacing:0.03em;">{_escape(e["type"])}</span>'
            f'{warn}</mark>'
        )
        cursor = end

    pieces.append(_escape(text[cursor:]))  # remaining plain text after the last entity
    body = "".join(pieces).replace("\n", "<br/>")
    return (
        '<div style="line-height:2.1;font-size:1.02rem;padding:14px 16px;'
        'border:1px solid rgba(128,128,128,0.25);border-radius:10px;">' + body + "</div>"
    )


def _escape(text: str) -> str:
    """Make text safe to drop into HTML (so a "<" in the document doesn't look like a tag)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _darken(hex_color: str) -> str:
    """Darken a "#RRGGBB" color, used for label text so it stays readable on a light highlight."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))  # hex pairs -> numbers 0-255
    r, g, b = (max(0, int(c * 0.55)) for c in (r, g, b))  # scale toward black
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Gold JSON export / persistence (also feeds the Annotation Evaluation page)
# ---------------------------------------------------------------------------


def build_gold_export(
    doc_name: str,
    doc_text: str,
    labels: list[str],
    method_id: str,
    entities: list[dict],
    run_meta: dict | None = None,
    source_meta: dict | None = None,
    classification: dict | None = None,
) -> dict:
    """Package one document's reviewed entities into a dict, ready to save as JSON.

    "gold_entities" = the final, kept spans (the actual training/eval data).
    "review_log" = every span including deleted ones, so we can later measure
    how much the human changed vs. what the model first suggested.
    """
    kept = [e for e in entities if e["status"] != "deleted"]
    return {
        "doc_name": doc_name,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "method": method_id,
        "labels": labels,
        "uncertainty": run_meta or {},
        "source": source_meta or {},
        "classification": classification or {},
        "text": doc_text,
        # Immutable model output, kept separate from the human-corrected gold.
        # This is what must be compared with hidden benchmark annotations.
        "model_predictions": [
            {
                "text": e.get("original_text", e["text"]),
                "type": e.get("original_type", e["type"]),
                "start": e.get("original_start", e["start"]),
                "end": e.get("original_end", e["end"]),
                "model_name": e.get("model_name"),
            }
            for e in entities if e.get("source") == "model"
        ],
        "gold_entities": [
            {
                "text": e["text"], "type": e["type"], "start": e["start"], "end": e["end"],
                "source": e["source"], "model_name": e.get("model_name"),
            }
            for e in kept
        ],
        "review_log": [
            {
                "text": e["text"], "type": e["type"], "start": e["start"], "end": e["end"],
                "source": e["source"], "model_name": e.get("model_name"), "status": e["status"],
                # Keeping the confidence next to the outcome is the whole
                # point: it lets us check afterwards whether low-confidence
                # rows really were the ones the reviewer changed.
                "confidence": e.get("confidence"),
                "conf_source": e.get("conf_source"),
                "voters": e.get("voters"),
            }
            for e in entities
        ],
    }


def save_gold_export(export: dict) -> Path:
    """Write a gold export dict to disk as a JSON file, and return its path."""
    GOLD_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in export["doc_name"])[:40]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = GOLD_EXPORT_DIR / f"{safe_name}_{timestamp}.json"
    path.write_text(json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_gold_exports() -> list[dict]:
    """Read every saved gold-export JSON file back into a list of dicts."""
    if not GOLD_EXPORT_DIR.exists():
        return []
    exports = []
    for path in sorted(GOLD_EXPORT_DIR.glob("*.json")):
        try:
            exports.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue  # skip a broken file instead of crashing the whole page
    return exports


def compute_annotation_metrics(export: dict) -> dict:
    """Turn one document's review_log into summary numbers for the Annotation Evaluation page."""
    log = export.get("review_log", export.get("entities", []))
    model_spans = [e for e in log if e["source"] == "model"]

    confirmed = sum(1 for e in model_spans if e["status"] == "confirmed")
    edited = sum(1 for e in model_spans if e["status"] == "edited")
    deleted = sum(1 for e in model_spans if e["status"] == "deleted")
    human_added = sum(1 for e in log if e["source"] == "human")

    n_model = len(model_spans)
    metrics = {
        "total_gold_entities": len(export.get("gold_entities", [])),
        "model_suggested": n_model,
        "confirmed_as_is": confirmed,
        "relabeled_by_human": edited,
        "deleted_by_human": deleted,
        "added_by_human": human_added,
        "acceptance_rate": round(confirmed / n_model, 3) if n_model else None,
        "edit_rate": round(edited / n_model, 3) if n_model else None,
        "deletion_rate": round(deleted / n_model, 3) if n_model else None,
    }
    metrics.update(compute_confidence_diagnostics(model_spans))
    return metrics


def compute_confidence_diagnostics(model_spans: list[dict], threshold: float = 0.75) -> dict:
    """Check whether the confidence score was actually worth computing.

    This is the honest test of the whole uncertainty feature. A confidence
    score is only useful if the spans it flags are the ones the human ends up
    changing. So we split the model's suggestions into flagged (below the
    threshold) and not flagged, and compare how often each group was touched.

    - `flag_precision`: of the rows we flagged, how many did the human change?
    - `flag_recall`: of the rows the human changed, how many had we flagged?
    - `lift`: how much more likely a flagged row was to be changed than an
      unflagged one. A lift near 1.0 means the score is not telling us
      anything and the extra LLM calls were wasted.
    """
    scored = [e for e in model_spans if isinstance(e.get("confidence"), (int, float))]
    if not scored:
        return {"confidence_available": False}

    def changed(e: dict) -> bool:
        return e.get("status") in ("edited", "deleted")

    flagged = [e for e in scored if e["confidence"] < threshold]
    unflagged = [e for e in scored if e["confidence"] >= threshold]
    changed_all = [e for e in scored if changed(e)]
    changed_flagged = [e for e in flagged if changed(e)]

    rate_flagged = (len(changed_flagged) / len(flagged)) if flagged else None
    rate_unflagged = (
        (sum(1 for e in unflagged if changed(e)) / len(unflagged)) if unflagged else None
    )

    lift = None
    if rate_flagged is not None and rate_unflagged:
        lift = round(rate_flagged / rate_unflagged, 2)
    elif rate_flagged and rate_unflagged == 0:
        lift = float("inf")  # every change was in the flagged group

    return {
        "confidence_available": True,
        "threshold": threshold,
        "n_scored": len(scored),
        "n_flagged": len(flagged),
        "share_flagged": round(len(flagged) / len(scored), 3),
        "n_changed": len(changed_all),
        "flag_precision": round(rate_flagged, 3) if rate_flagged is not None else None,
        "flag_recall": (
            round(len(changed_flagged) / len(changed_all), 3) if changed_all else None
        ),
        "change_rate_flagged": round(rate_flagged, 3) if rate_flagged is not None else None,
        "change_rate_unflagged": round(rate_unflagged, 3) if rate_unflagged is not None else None,
        "lift": lift,
        "mean_confidence": round(sum(e["confidence"] for e in scored) / len(scored), 3),
    }
