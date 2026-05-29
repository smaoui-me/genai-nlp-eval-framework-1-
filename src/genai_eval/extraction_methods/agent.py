from pathlib import Path

from genai_eval.llm_client import call_llm, get_model_name
from genai_eval.extraction_methods.base import ExtractionMethod
from genai_eval.json_utils import parse_json_response
from genai_eval.prompts import format_prompt, load_prompt_template
from genai_eval.config_loader import load_config

DEFAULT_CONFIG_PATH = Path("configs/agent.yaml")


def run_step1(prompt_template: str, text: str, llm_params: dict) -> dict:
    """Step 1: Extract evidence snippets from the ticket text."""
    prompt = format_prompt(prompt_template, ticket_text=text)
    raw = call_llm(prompt, **llm_params)
    parsed, json_valid, json_error = parse_json_response(raw)
    return {
        "raw_response": raw,
        "parsed_output": parsed if isinstance(parsed, dict) else {},
        "json_valid": json_valid,
        "json_error": json_error,
    }


def run_step2(
    prompt_template: str,
    step1_result: dict,
    allowed_labels: dict,
    llm_params: dict,
) -> dict:
    """Step 2: Map extracted evidence snippets to allowed labels."""
    topic_evidence = step1_result.get("parsed_output", {}).get("topic_evidence", [])
    topic_str = (
        ", ".join(topic_evidence)
        if isinstance(topic_evidence, list)
        else str(topic_evidence)
    )
    candidate_tags = allowed_labels.get("candidate_tags", [])
    prompt = format_prompt(
        prompt_template,
        issue_type_evidence=step1_result.get("parsed_output", {}).get("issue_type_evidence", ""),
        queue_evidence=step1_result.get("parsed_output", {}).get("queue_evidence", ""),
        topic_evidence=topic_str,
        allowed_types=allowed_labels["types"],
        allowed_queues=allowed_labels["queues"],
        candidate_tags=candidate_tags,
    )
    raw = call_llm(prompt, **llm_params)
    parsed, json_valid, json_error = parse_json_response(raw)
    return {
        "raw_response": raw,
        "parsed_output": parsed if isinstance(parsed, dict) else {},
        "json_valid": json_valid,
        "json_error": json_error,
    }


class AgentTicketExtraction(ExtractionMethod):
    """Two-step agent extraction: evidence identification → label mapping."""

    name = "agent_two_step"

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config = load_config(config_path)
        prompts_cfg = self.config.get("prompts", {})
        self.step1_template = load_prompt_template(Path(prompts_cfg["step1"]))
        self.step2_template = load_prompt_template(Path(prompts_cfg["step2"]))
        self.llm_params = self.config.get("llm", {})
        self.top_k_tags = self.config.get("extraction", {}).get("top_k_tags")
        self.model_name = get_model_name()

    def extract_record(
        self,
        text: str,
        allowed_labels: dict,
        context: dict | None = None,
    ) -> dict:
        candidate_tags = (context or {}).get("candidate_tags", allowed_labels.get("tags", []))
        step1_result = run_step1(self.step1_template, text, self.llm_params)
        step2_labels = dict(allowed_labels)
        step2_labels["candidate_tags"] = candidate_tags
        prediction = run_step2(
            self.step2_template, step1_result, step2_labels, self.llm_params
        )
        validated_output, validation = self.validate_against_labels(
            prediction=prediction.get("parsed_output"),
            allowed_labels=allowed_labels,
            candidate_tags=candidate_tags,
            top_k_tags=self.top_k_tags,
        )
        return {
            "raw_responses": {
                "step1": step1_result["raw_response"],
                "step2": prediction["raw_response"],
            },
            "parsed_output": {
                "step1": step1_result["parsed_output"],
                "step2": prediction["parsed_output"],
            },
            "validated_output": validated_output,
            "json_validity": {
                "step1_json_valid": step1_result["json_valid"],
                "step1_json_error": step1_result["json_error"],
                "step2_json_valid": prediction["json_valid"],
                "step2_json_error": prediction["json_error"],
                "all_json_valid": step1_result["json_valid"] and prediction["json_valid"],
            },
            "validation": validation,
            "prompt_inputs": {"candidate_tags": candidate_tags},
        }

    def extract(self, text: str, allowed_labels: dict, context: dict | None = None) -> dict:
        return self.extract_record(text, allowed_labels, context=context)["validated_output"]
