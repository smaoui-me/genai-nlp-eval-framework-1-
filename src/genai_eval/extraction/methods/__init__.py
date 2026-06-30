from genai_eval.extraction.methods.few_shot import FewShotExtractionMethod
from genai_eval.extraction.methods.zero_shot import ZeroShotExtractionMethod

METHOD_REGISTRY = {
    "zero_shot": ZeroShotExtractionMethod,
    "few_shot": FewShotExtractionMethod,
}

__all__ = ["METHOD_REGISTRY", "FewShotExtractionMethod", "ZeroShotExtractionMethod"]
