"""
few_shot.py

Few-shot ticket extraction. Identical to zero-shot but uses a prompt template
that includes demonstration examples. All extraction logic is inherited from
ZeroShotTicketExtraction; only the default config path differs.
"""

from pathlib import Path

from genai_eval.extraction_methods.zero_shot import ZeroShotTicketExtraction

DEFAULT_CONFIG_PATH = Path("configs/few_shot.yaml")


class FewShotTicketExtraction(ZeroShotTicketExtraction):
    """Few-shot extraction: same pipeline as zero-shot, prompt includes examples."""

    name = "few_shot"

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        super().__init__(config_path)
