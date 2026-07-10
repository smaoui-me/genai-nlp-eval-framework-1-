"""The default entity labels, and a consistent color for each one so
highlighted text always looks the same everywhere in the app."""

from __future__ import annotations

# Pre-selected when someone opens the Annotate page — they can add/remove labels there.
DEFAULT_LABELS = ["location", "person", "organization", "product", "event", "building"]

# A fixed color (hex code, #RRGGBB = red/green/blue) for each common label.
_PALETTE = {
    "location": "#60A5FA",
    "person": "#F472B6",
    "organization": "#34D399",
    "product": "#FBBF24",
    "event": "#A78BFA",
    "building": "#F87171",
    "art": "#22D3EE",
    "other": "#9CA3AF",
}

# Backup colors for custom labels that aren't in the palette above.
_FALLBACK_PALETTE = ["#60A5FA", "#F472B6", "#34D399", "#FBBF24", "#A78BFA", "#F87171", "#22D3EE", "#9CA3AF", "#FB923C", "#4ADE80"]


def color_for_label(label: str, labels: list[str]) -> str:
    """Return a hex color for a label. Known labels always get the same
    color; unknown/custom labels get one picked from the fallback list."""
    key = label.strip().lower()
    if key in _PALETTE:
        return _PALETTE[key]
    # Use the label's position in `labels` to pick a stable fallback color,
    # or hash() as a last resort if we don't even have that.
    index = labels.index(label) if label in labels else hash(key) % len(_FALLBACK_PALETTE)
    return _FALLBACK_PALETTE[index % len(_FALLBACK_PALETTE)]
