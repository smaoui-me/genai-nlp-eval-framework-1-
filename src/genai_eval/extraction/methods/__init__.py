from genai_eval.extraction.methods.zero_shot_freeform import ZeroShotFreeformExtractionMethod
from genai_eval.extraction.methods.zero_shot_structured import ZeroShotStructuredExtractionMethod
from genai_eval.extraction.methods.few_shot import FewShotExtractionMethod
METHOD_REGISTRY = {
    "zero_shot_structured": ZeroShotStructuredExtractionMethod,
    "zero_shot_freeform": ZeroShotFreeformExtractionMethod,
    "few_shot": FewShotExtractionMethod,
}

__all__ = ["METHOD_REGISTRY", "ZeroShotStructuredExtractionMethod", "ZeroShotFreeformExtractionMethod", "FewShotExtractionMethod"]
