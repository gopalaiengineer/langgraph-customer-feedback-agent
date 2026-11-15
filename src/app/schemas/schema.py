
from typing import Literal, TypedDict, Optional

from pydantic import BaseModel, Field, model_validator


class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(...,
        description="Overall customer feedback sentiment"
    )
    reason: str = Field(
        description="Short explanation for the classification"
    )


class FeedbackAnalysis(BaseModel):
    issue_category: Literal[
        "product_quality",
        "delivery",
        "billing",
        "technical_support",
        "customer_service",
        "refund",
        "other",
    ] = Field(description="Category of the customer's issue")
    urgency: Literal["low", "medium", "high", "critical"] = Field(
        description="How quickly the issue needs attention based on the feedback"
    )
    priority: Literal["P3", "P2", "P1"] = Field(
        description="P3 for minor dissatisfaction, P2 for high-impact issues, "
        "or P1 for critical complaints"
    )
    recommended_action: str = Field(
        description="Recommended next step; do not claim an action has happened"
    )
    escalation_level: Literal[
        "none",
        "support",
        "support_and_managers",
    ] = Field(description="Internal escalation recipients required")
    escalation_reason: str = Field(
        description="Evidence-based reason for the selected escalation level"
    )
    issue_type: Literal[
        "product_quality",
        "delivery",
        "billing",
        "technical_support",
        "customer_service",
        "refund",
        "other",
    ]
    tone: Literal[
        "angry",
        "frustrated",
        "disappointed",
        "concerned",
        "neutral",
    ]
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Impact of the issue on the customer, independent of response time"
    )
    summary: str = Field(
        description="Concise summary of the customer's problem"
    )

    @model_validator(mode="after")
    def validate_priority_escalation(self):
        expected = {
            "P3": ("low", "none"),
            "P2": ("high", "support"),
            "P1": ("critical", "support_and_managers"),
        }[self.priority]
        if (self.urgency, self.escalation_level) != expected:
            raise ValueError(
                "Priority, urgency, and escalation_level must follow the "
                "P3/low/none, P2/high/support, or "
                "P1/critical/support_and_managers policy."
            )
        return self


class EmailDraft(BaseModel):
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Complete professional email body")


class EmailDraftReview(TypedDict):
    subject: str
    body: str
    score: int


class FeedbackState(TypedDict):
    customer_name: str
    customer_email: str
    feedback: str
    sentiment: str
    sentiment_reason: Optional[str]
    analysis: Optional[dict]
    support_ticket: Optional[dict]
    internal_notification_status: Optional[str]
    email_subject: Optional[str]
    email_body: Optional[str]
    email_status: Optional[str]
    approval_status: Optional[
        Literal["approved", "needs_evaluation", "stopped"]
    ]
    email_clarification: Optional[str]
    email_review_attempts: int
    email_draft_history: list[EmailDraftReview]
    email_selected_score: Optional[int]

