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
    st.session_state.setdefault("exported_docs", [])
    # Bumped every time entities change from a button click (not from typing
    # in the table itself). We use it inside the data_editor's key so the
    # table always redraws with the latest data — see pages/1_annotate.py.
    st.session_state.setdefault("entities_version", 0)


def bump_entities_version() -> None:
    """Call this after changing st.session_state.entities from a button, so
    the review table below is forced to redraw with the new data."""
    st.session_state.entities_version += 1


def new_entity_id() -> str:
    """A short random ID, unique enough to tell entities apart in one document."""
    return uuid.uuid4().hex[:8]


def entities_to_dataframe(entities: list[dict]) -> pd.DataFrame:
    """Turn our list of entity dicts into a table (DataFrame) for st.data_editor."""
    visible = [e for e in entities if e["status"] != "deleted"]
    if not visible:
        # An empty pd.DataFrame(columns=[...]) gives every column "object"
        # dtype, but the "confirmed"/"delete" checkbox columns and
        # "start"/"end" number columns below need real bool/int dtypes to
        # match st.data_editor's column_config — a mismatch here can crash
        # the app when it's serialized for display, so we set them by hand.
        return pd.DataFrame({
            "id": pd.Series(dtype="object"),
            "text": pd.Series(dtype="object"),
            "type": pd.Series(dtype="object"),
            "start": pd.Series(dtype="int64"),
            "end": pd.Series(dtype="int64"),
            "source": pd.Series(dtype="object"),
            "status": pd.Series(dtype="object"),
            "confirmed": pd.Series(dtype="bool"),
            "delete": pd.Series(dtype="bool"),
        })

    df = pd.DataFrame(visible)

    # A friendly "who suggested this" column: the model name if the LLM
    # suggested it, otherwise "human".
    df["source"] = df.apply(
        lambda row: row["model_name"] if row["source"] == "model" and row.get("model_name") else row["source"],
        axis=1,
    )

    # A checkbox column showing/controlling whether each span has been
    # reviewed. "edited" also counts as reviewed (you can't relabel
    # something without looking at it).
    df["confirmed"] = df["status"].isin(["confirmed", "edited"])
    df["delete"] = False

    return df[["id", "text", "type", "start", "end", "source", "status", "confirmed", "delete"]]


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


def render_highlighted_html(text: str, entities: list[dict], labels: list[str]) -> str:
    """Build an HTML snippet showing `text` with each entity highlighted in its label's color."""
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
        pieces.append(
            f'<mark style="background:{color}33;border:{border} {color};border-radius:5px;'
            f'padding:1px 4px;margin:0 1px;">{_escape(text[start:end])}'
            f'<span style="font-size:0.68em;font-weight:700;color:{_darken(color)};'
            f'text-transform:uppercase;margin-left:5px;letter-spacing:0.03em;">{_escape(e["type"])}</span></mark>'
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


def build_gold_export(doc_name: str, doc_text: str, labels: list[str], method_id: str, entities: list[dict]) -> dict:
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
        "text": doc_text,
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
    return {
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
