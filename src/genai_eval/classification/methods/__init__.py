"""
Classification-with-evidence method registry.
"""

from genai_eval.classification.methods.agent import AgentHierarchyAwareClassification
from genai_eval.classification.methods.base import EvidenceClassificationMethod
from genai_eval.classification.methods.few_shot import FewShotEvidenceClassification
from genai_eval.classification.methods.zero_shot import ZeroShotEvidenceClassification


METHOD_REGISTRY = {
    ZeroShotEvidenceClassification.name: ZeroShotEvidenceClassification,
    AgentHierarchyAwareClassification.name: AgentHierarchyAwareClassification,
    "agent": AgentHierarchyAwareClassification,
    FewShotEvidenceClassification.name: FewShotEvidenceClassification,
}

