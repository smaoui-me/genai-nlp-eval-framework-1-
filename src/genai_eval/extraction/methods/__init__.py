from genai_eval.extraction.methods.few_shot_freeform import FewShotFreeformExtractionMethod
from genai_eval.extraction.methods.few_shot_structured import FewShotStructuredExtractionMethod
from genai_eval.extraction.methods.zero_shot_freeform import ZeroShotFreeformExtractionMethod
from genai_eval.extraction.methods.zero_shot_structured import ZeroShotStructuredExtractionMethod

from genai_eval.extraction.methods.agent_verify_extraction import AgentVerifyExtractionMethod
METHOD_REGISTRY = {
    "zero_shot_structured": ZeroShotStructuredExtractionMethod,
    "zero_shot_freeform": ZeroShotFreeformExtractionMethod,
    "few_shot_structured": FewShotStructuredExtractionMethod,
    "few_shot_freeform": FewShotFreeformExtractionMethod,
    "agent_verify_extraction": AgentVerifyExtractionMethod,
}

__all__ = [
    "METHOD_REGISTRY",
    "ZeroShotStructuredExtractionMethod",
    "ZeroShotFreeformExtractionMethod",
    "FewShotStructuredExtractionMethod",
    "FewShotFreeformExtractionMethod",
, "AgentVerifyExtractionMethod"]
