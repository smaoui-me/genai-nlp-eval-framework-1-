"""Build editable extraction prompts from a user-facing annotation schema."""

from __future__ import annotations

import hashlib


LABEL_DEFINITIONS = {
    "Method": "An algorithm, model, architecture, system, loss, optimization procedure, or technical approach.",
    "Task": "A research problem, prediction objective, or application task.",
    "Metric": "A named evaluation measure or criterion, such as accuracy, F1, BLEU, runtime, or mean squared error.",
    "Dataset": "A named dataset, corpus, benchmark, data collection, language resource, or input data source used for training or evaluation.",
}


def display_label(label: str) -> str:
    """Use the clearer public name while retaining compatibility with SciREX."""
    return "Dataset" if label.casefold() == "material" else label


def canonical_label(label: str) -> str:
    """Normalize the old SciREX name and the user-facing name to one class."""
    return "dataset" if label.casefold() in {"material", "dataset"} else label.casefold()


def suggest_prompt(
    labels: list[str],
    dataset_name: str | None = None,
    structured: bool = True,
    label_definitions: dict[str, str] | None = None,
    domain_guidance: str | None = None,
) -> str:
    """Create a deterministic prompt template that users can inspect and edit."""
    visible = list(dict.fromkeys(display_label(label) for label in labels))
    context = f" for the {dataset_name} dataset" if dataset_name else ""
    configured_definitions = {
        display_label(str(label)): str(description)
        for label, description in (label_definitions or {}).items()
    }
    definitions = []
    for label in visible:
        description = configured_definitions.get(label) or LABEL_DEFINITIONS.get(
            label, f"A mention that satisfies the project's definition of {label}."
        )
        definitions.append(f"- {label}: {description}")
    guidance = (
        f"\nDomain-specific guidance:\n- {domain_guidance.strip()}\n"
        if domain_guidance and domain_guidance.strip() else ""
    )
    output = (
        'Return ONLY valid JSON: {{"entities": [{{"text": "exact token text", '
        '"type": "one allowed label", "start": 0, "end": 0}}]}}.'
        if structured else
        "Return one entity per line as: exact text | label | inclusive start token | inclusive end token. Return NONE if empty."
    )
    return f"""You are an annotation assistant{context}. Extract every explicit entity mention that matches the annotation schema.

Annotation schema:
{chr(10).join(definitions)}
{guidance}

Rules:
- Use only the labels above.
- Extract the smallest complete meaningful span, including necessary words such as model, system, task, score, or dataset when they are part of the mention.
- Include names, abbreviations, repeated mentions, and clear descriptive noun phrases.
- Do not infer entities that are not explicitly written.
- Do not label generic technical words unless they clearly satisfy a schema definition.
- For Dataset, do not label algorithms, models, hardware, mathematical objects, functions, constraints, or ordinary substances.
- `start` and `end` are inclusive indices in the token list.
- The returned text must exactly match the indexed tokens.

Sentence:
{{sentence}}

Token indices:
{{indexed_tokens}}

{output}
If nothing matches, return {{{{"entities": []}}}}.
"""


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
