"""Prompt suggestions are editable templates, not hidden LLM calls."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.extraction_methods import _build_prompt
from utils.prompt_builder import canonical_label, prompt_hash, suggest_prompt


prompt = suggest_prompt(["Method", "Task", "Metric", "Dataset"], "SciREX")
assert "{sentence}" in prompt
assert "{indexed_tokens}" in prompt
assert "Dataset" in prompt and "ordinary substances" in prompt
rendered = _build_prompt("few_shot_structured", [], "MNIST is used.", "0: MNIST", prompt)
assert "MNIST is used." in rendered and "0: MNIST" in rendered
assert canonical_label("Material") == canonical_label("Dataset") == "dataset"
assert prompt_hash(prompt) == prompt_hash(prompt)
print("Prompt builder checks passed.")
