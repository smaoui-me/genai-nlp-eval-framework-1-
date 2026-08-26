"""Optional, evidence-carrying ticket routing for the annotation workflow."""

from __future__ import annotations

import json
import re

from utils.llm_client import call_llm_full
from utils.model_registry import ModelChoice


def suggest_routing_prompt(
    departments: list[str], use_case_name: str = "support tickets", guidance: str = ""
) -> str:
    allowed = "\n".join(f"- {department}" for department in departments)
    extra = f"\nDomain guidance: {guidance.strip()}" if guidance.strip() else ""
    return f"""You route {use_case_name} to exactly one allowed department.

Allowed departments:
{allowed}
{extra}

Rules:
- Select only one department from the allowed list.
- Base the decision on the primary action required to resolve the ticket.
- Do not invent missing facts.
- `evidence` must be a short exact quote from the ticket.
- Return only valid JSON with this shape:
  {{"department": "allowed department", "reason": "brief explanation", "evidence": "exact quote"}}

Ticket:
{{ticket_text}}
"""


def _parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Routing response did not contain a JSON object") from None
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Routing response was not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Routing response must be a JSON object")
    return value


def validate_routing(payload: dict, departments: list[str], ticket_text: str) -> dict:
    by_normalized = {department.casefold(): department for department in departments}
    requested = str(payload.get("department", "")).strip()
    department = by_normalized.get(requested.casefold())
    if not department:
        raise ValueError(f"Model returned a department outside the allowed list: {requested!r}")
    reason = str(payload.get("reason", "")).strip()
    evidence = str(payload.get("evidence", "")).strip()
    evidence_valid = bool(evidence and evidence.casefold() in ticket_text.casefold())
    return {
        "department": department,
        "reason": reason,
        "evidence": evidence if evidence_valid else "",
        "evidence_valid": evidence_valid,
    }


def route_ticket(
    ticket_text: str,
    departments: list[str],
    prompt_template: str,
    choice: ModelChoice,
    llm_params: dict | None = None,
) -> dict:
    if not departments:
        raise ValueError("At least one destination department is required")
    if "{ticket_text}" not in prompt_template:
        raise ValueError("Routing prompt must contain the {ticket_text} placeholder")
    response = call_llm_full(
        prompt_template.replace("{ticket_text}", ticket_text),
        choice=choice,
        **(llm_params or {}),
    )
    result = validate_routing(_parse_json_object(response.text), departments, ticket_text)
    result.update({
        "classifier": "llm",
        "model": response.model_id,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": response.total_tokens,
        "usage_reported": response.usage_reported,
    })
    return result
