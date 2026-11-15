from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_LLM_GUARDRAIL_RESPONSE = (
    "I can help only with the Customer Feedback Management System, "
    "including customer feedback analysis, complaints, sentiment analysis, "
    "support tickets, priority classification, escalation, and customer "
    "email workflows. Please provide a related request."
)

CUSTOMER_FACING_LLM_GUARDRAIL_RESPONSE = (
    "I can help only with customer feedback management-related queries. "
    "Please provide a related request."
)


class CustomGuardrailResult(BaseModel):
    allowed: bool
    reason: str
    matched_keywords: list[str]
    restricted_matches: list[str]
    category: str


class LLMGuardrailResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed: bool
    category: str
    reason: str
    risk_flags: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    safe_response: str | None = None

    @model_validator(mode="after")
    def validate_category_and_consistency(self) -> "LLMGuardrailResult":
        valid_categories = {
            "customer_feedback",
            "customer_complaint",
            "sentiment_analysis",
            "priority_classification",
            "ticket_management",
            "email_workflow",
            "project_operations",
            "out_of_scope",
            "prompt_injection",
            "sensitive_data_request",
        }
        if self.category not in valid_categories:
            raise ValueError(f"Unsupported category: {self.category}")
        if self.allowed and self.category in {
            "out_of_scope",
            "prompt_injection",
            "sensitive_data_request",
        }:
            raise ValueError("Rejected categories cannot be marked allowed.")
        if not self.allowed and self.safe_response is None:
            self.safe_response = DEFAULT_LLM_GUARDRAIL_RESPONSE
        return self
