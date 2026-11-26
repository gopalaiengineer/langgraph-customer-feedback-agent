from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EvaluationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_name: str
    customer_email: str
    feedback: str = Field(min_length=1, max_length=5000)


class ExpectedOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentiment: Literal["positive", "negative"]
    category: Literal[
        "product_quality",
        "delivery",
        "billing",
        "technical_support",
        "customer_service",
        "refund",
        "other",
    ] | None
    intent: str
    priority: Literal["P3", "P2", "P1"] | None = None
    actions: list[
        Literal[
            "prepare_positive_email",
            "prepare_negative_email",
            "create_support_ticket",
            "send_internal_notification",
        ]
    ]


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    input: EvaluationInput
    expected: ExpectedOutput


class AgentOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: dict[str, Any]
    nodes: list[str]
    guardrails: dict[str, bool]
    simulated_emails: int = 0
    simulated_tickets: int = 0


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    execution_status: str
    execution_error: str | None = None
    prediction_errors: list[str] = Field(default_factory=list)
    expected: dict[str, Any]
    predicted: dict[str, Any] | None = None
    nodes: list[str] = Field(default_factory=list)
    guardrails: dict[str, bool] = Field(default_factory=dict)
    deterministic_scores: dict[str, float | None] = Field(default_factory=dict)
    judge_scores: dict[str, int] | None = None
    judge_error: str | None = None
    simulated_emails: int = 0
    simulated_tickets: int = 0


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_version: str = "1"
    agent_version: str
    dataset: str
    started_at: str
    finished_at: str
    summary: dict[str, Any]
    results: list[CaseResult]
