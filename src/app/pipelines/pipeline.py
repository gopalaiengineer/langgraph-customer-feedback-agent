import re
from functools import lru_cache
from pathlib import Path

from langgraph.types import Command
from app.agents.agent import (
    agent_manager,
    draft_ticket_resolution_email,
    send_ticket_resolution_email,
)
from app.guardrail import (
    ALLOWED_KEYWORDS,
    REVIEW_ALLOWED_KEYWORDS,
    custom_guardrail,
    llm_guardrail,
)
from app.schemas.schema import FeedbackState
from app.database.repositories import SupportTicketRepository

_ticket_repository = SupportTicketRepository()
_EMAIL_PATTERN = re.compile(
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
)


@lru_cache(maxsize=1)
def _get_graph():
    return agent_manager()


def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# ---------- RUN THE WORKFLOW ----------

def customer_feedback_Analysis(
    feedback: str,
    customer_name: str,
    customer_email: str,
    thread_id: str,
) -> FeedbackState:
    validation = custom_guardrail(feedback)
    if not validation.allowed:
        raise ValueError(
            f"Customer feedback rejected ({validation.category}): "
            f"{validation.reason}"
        )

    llm_validation = llm_guardrail(feedback, audience="customer")
    if not llm_validation.allowed:
        raise ValueError(
            llm_validation.safe_response or llm_validation.reason
        )

    graph = _get_graph()
    graph_path = Path("customer_feedback_graph.png")
    if not graph_path.exists():
        graph_png = graph.get_graph().draw_mermaid_png()
        graph_path.write_bytes(graph_png)
    else:
        print("Graph already exists. Skipping generation.")

    initial_state: FeedbackState = {
        "customer_name": customer_name,
        "customer_email": customer_email,
        "feedback": feedback,
        "sentiment": None,
        "sentiment_reason": None,
        "analysis": None,
        "support_ticket": None,
        "internal_notification_status": None,
        "email_subject": None,
        "email_body": None,
        "email_status": "awaiting_approval",
        "approval_status": None,
        "email_clarification": None,
        "email_review_attempts": 0,
        "email_draft_history": [],
        "email_selected_score": None,
    }

    config = _thread_config(thread_id)
    graph.invoke(initial_state, config=config)
    return graph.get_state(config).values


def resume_customer_feedback_Analysis(
    thread_id: str,
    approval_status: str | None = None,
    score: int | None = None,
    clarification: str = "",
    final_decision: str | None = None,
    selected_draft_index: int | None = None,
) -> FeedbackState:
    graph = _get_graph()
    config = _thread_config(thread_id)
    if final_decision is not None:
        final_decision = final_decision.strip().lower()
        if final_decision not in {"send", "stop"}:
            raise ValueError("Final decision must be 'send' or 'stop'.")

        current_state = graph.get_state(config).values
        if current_state.get("email_status") != "awaiting_final_selection":
            raise ValueError("The workflow is not waiting for a final selection.")
        history = current_state.get("email_draft_history", [])
        if final_decision == "send" and (
            type(selected_draft_index) is not int
            or not 0 <= selected_draft_index < len(history)
        ):
            raise ValueError("Select a valid reviewed email draft.")

        resume_value = {
            "final_decision": final_decision,
            "selected_draft_index": selected_draft_index,
        }
    else:
        if approval_status is None:
            raise ValueError(
                "Select Accepted, Needs Evaluation, or Stop as the approval status."
            )
        approval_status = approval_status.strip().lower()
        approval_status = {
            "accepted": "approved",
            "needs evaluation": "needs_evaluation",
            "stop": "stopped",
        }.get(approval_status)
        if approval_status not in {"approved", "needs_evaluation", "stopped"}:
            raise ValueError(
                "Select Accepted, Needs Evaluation, or Stop as the approval status."
            )

        if approval_status != "stopped" and (
            type(score) is not int or not 1 <= score <= 10
        ):
            raise ValueError("Enter a score from 1 to 10 for this email.")
        if (
            approval_status in {"needs_evaluation", "stopped"}
            and not clarification.strip()
        ):
            raise ValueError(
                "Please provide a clarification for Needs Evaluation or Stop."
            )

        if approval_status == "needs_evaluation":
            validation = custom_guardrail(
                clarification,
                allowed_keywords=[
                    *ALLOWED_KEYWORDS,
                    *REVIEW_ALLOWED_KEYWORDS,
                ],
            )
            if not validation.allowed:
                raise ValueError(
                    f"Email review clarification rejected "
                    f"({validation.category}): {validation.reason}"
                )

        resume_value = {"approval_status": approval_status}
        if approval_status != "stopped":
            resume_value.update(
                {
                    "score": score,
                    "clarification": clarification,
                }
            )
        else:
            resume_value["clarification"] = clarification

    graph.invoke(
        Command(resume=resume_value),
        config=config,
    )
    return graph.get_state(config).values


def list_support_tickets() -> list[dict]:
    return _ticket_repository.list_all()


def build_ticket_resolution_email(
    ticket_id: str,
    ticket_data: dict,
    resolution_reason: str,
) -> tuple[str, str, str]:
    customer_name = (
        str(ticket_data.get("customer_name", "")).strip() or "Customer"
    )
    customer_email = _ticket_customer_email(ticket_data)
    summary = str(ticket_data.get("summary", "")).strip()
    if not summary:
        raise ValueError("This ticket has no summary to include in the email.")

    draft = draft_ticket_resolution_email(
        ticket_id,
        customer_name,
        summary,
        resolution_reason,
    )
    return customer_email, draft.subject, draft.body


def _ticket_customer_email(ticket_data: dict) -> str:
    customer_email = str(ticket_data.get("customer_email", "")).strip()
    if not customer_email or not _EMAIL_PATTERN.fullmatch(customer_email):
        raise ValueError(
            "This ticket has no valid registered customer email address."
        )
    return customer_email


def resolve_support_ticket(
    ticket_id: str,
    resolution_reason: str,
    admin_username: str,
) -> dict:
    reason = resolution_reason.strip()
    if not reason:
        raise ValueError("Enter a resolution reason or clarification.")
    if len(reason) > 4000:
        raise ValueError("Resolution details must be 4,000 characters or fewer.")

    ticket = _ticket_repository.get_by_ticket_id(ticket_id)
    if ticket is None:
        raise LookupError("The requested support ticket could not be found.")

    data = ticket["ticket_data"]
    if data.get("status") in {"resolved", "rejected"}:
        raise ValueError(
            f"This support ticket has already been {data['status']}."
        )
    customer_email = _ticket_customer_email(data)
    pending = data.get("resolution_pending")
    if pending is not None:
        if pending.get("reason") != reason:
            raise ValueError(
                "A different resolution is already pending for this ticket."
            )
        subject = pending["email_subject"]
        body = pending["email_body"]
    else:
        customer_email, subject, body = build_ticket_resolution_email(
            ticket_id,
            data,
            reason,
        )

    pending = _ticket_repository.begin_resolution(
        ticket["thread_id"],
        reason,
        subject,
        body,
    )

    send_ticket_resolution_email(
        ticket["thread_id"],
        ticket_id,
        customer_email,
        pending["email_subject"],
        pending["email_body"],
    )
    return _ticket_repository.mark_resolved(
        ticket["thread_id"],
        pending["reason"],
        admin_username,
        pending["email_subject"],
        pending["email_body"],
    )


def reject_support_ticket(ticket_id: str) -> dict:
    ticket = _ticket_repository.get_by_ticket_id(ticket_id)
    if ticket is None:
        raise LookupError("The requested support ticket could not be found.")
    return _ticket_repository.mark_rejected(ticket["thread_id"])