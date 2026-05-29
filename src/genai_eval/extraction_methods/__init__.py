from genai_eval.extraction_methods.agent import AgentTicketExtraction
from genai_eval.extraction_methods.zero_shot import ZeroShotTicketExtraction


METHOD_REGISTRY = {
    ZeroShotTicketExtraction.name: ZeroShotTicketExtraction,
    AgentTicketExtraction.name: AgentTicketExtraction,
    "agent": AgentTicketExtraction,
}
