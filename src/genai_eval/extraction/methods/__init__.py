from genai_eval.extraction.methods.zero_shot_freeform import ZeroShotFreeformExtractionMethod
from genai_eval.extraction.methods.zero_shot_structured import ZeroShotStructuredExtractionMethod

METHOD_REGISTRY = {
    "zero_shot_structured": ZeroShotStructuredExtractionMethod,
    "zero_shot_freeform": ZeroShotFreeformExtractionMethod,
}

__all__ = ["METHOD_REGISTRY", "ZeroShotStructuredExtractionMethod", "ZeroShotFreeformExtractionMethod"]
