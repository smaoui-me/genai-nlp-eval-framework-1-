from genai_eval.extraction.methods.zero_shot import ZeroShotExtractionMethod

METHOD_REGISTRY = {
    "zero_shot": ZeroShotExtractionMethod,
}

__all__ = ["METHOD_REGISTRY", "ZeroShotExtractionMethod"]
