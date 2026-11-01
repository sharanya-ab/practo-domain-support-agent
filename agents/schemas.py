from pydantic import BaseModel, Field
class SupportResponse(BaseModel):
    answer: str
    intent: str = "policy"
    source_documents: list[str] = Field(default_factory=list)
    escalation_recommended: bool = False
    safety_notice: str = "Administrative support only; not medical advice."
