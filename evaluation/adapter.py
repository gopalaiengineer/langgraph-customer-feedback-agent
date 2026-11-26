from __future__ import annotations

import os
import threading
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

from .metrics import intent_from_state
from .models import AgentOutput, EvaluationCase


class LangGraphAgentAdapter:
    """Run the real compiled graph with memory checkpointing and fake side effects.

    Model calls are deliberately opt-in. The graph shares module-level integrations,
    so this adapter is single-threaded and patches them only for its lifetime.
    """

    thread_safe = False

    def __init__(self, *, allow_model_calls: bool, run_id: str) -> None:
        if not allow_model_calls:
            raise ValueError(
                "The graph uses external LLM providers; pass --allow-model-calls "
                "to explicitly permit synthetic evaluation prompts to be sent."
            )
        self.run_id = run_id
        self.simulated_emails = 0
        self.simulated_tickets: dict[str, dict[str, Any]] = {}
        self._delivery_records: dict[str, tuple[str, str | None]] = {}
        self._delivery_lock = threading.Lock()
        self._stack = ExitStack()
        self._entered = False

    def __enter__(self) -> "LangGraphAgentAdapter":
        if self._entered:
            raise RuntimeError("The evaluation adapter cannot be entered twice.")

        self._stack.enter_context(
            patch.dict(
                os.environ,
                {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"},
            )
        )
        from langgraph.checkpoint.memory import MemorySaver
        from app.agents import agent

        self._agent = agent

        def claim(key: str):
            with self._delivery_lock:
                previous = self._delivery_records.get(key)
                if previous is not None:
                    return False, previous
                self._delivery_records[key] = ("sending", None)
                return True, None

        def mark_sent(key: str) -> None:
            with self._delivery_lock:
                self._delivery_records[key] = ("sent", None)

        def mark_failed(key: str, error: str) -> None:
            with self._delivery_lock:
                self._delivery_records[key] = ("failed", error)

        def create_or_get(thread_id: str, ticket: dict[str, Any]):
            return self.simulated_tickets.setdefault(thread_id, ticket)

        class FakeSMTP:
            def __init__(inner_self, *_args: Any, **_kwargs: Any) -> None:
                pass

            def __enter__(inner_self):
                return inner_self

            def __exit__(inner_self, *_args: Any) -> bool:
                return False

            def ehlo(inner_self) -> None:
                return None

            def starttls(inner_self, **_kwargs: Any) -> None:
                return None

            def login(inner_self, *_args: Any) -> None:
                return None

            def send_message(inner_self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
                self.simulated_emails += 1
                return {}

        for target, attribute, replacement in (
            (agent, "get_checkpointer", lambda: MemorySaver()),
            (agent._delivery_repository, "claim", claim),
            (agent._delivery_repository, "mark_sent", mark_sent),
            (agent._delivery_repository, "mark_failed", mark_failed),
            (agent._ticket_repository, "create_or_get", create_or_get),
            (agent.smtplib, "SMTP", FakeSMTP),
        ):
            self._stack.enter_context(patch.object(target, attribute, replacement))

        self.graph = agent.agent_manager()
        self._entered = True
        return self

    def invoke(self, case: EvaluationCase) -> AgentOutput:
        if not self._entered:
            raise RuntimeError("Use the adapter inside a context manager.")

        from app.guardrail import custom_guardrail, llm_guardrail

        emails_before = self.simulated_emails
        tickets_before = len(self.simulated_tickets)
        feedback = case.input.feedback
        custom = custom_guardrail(feedback)
        guardrails = {"custom": custom.allowed, "llm": False}
        if not custom.allowed:
            return AgentOutput(
                state={},
                nodes=[],
                guardrails=guardrails,
                simulated_emails=self.simulated_emails - emails_before,
                simulated_tickets=len(self.simulated_tickets) - tickets_before,
            )

        llm = llm_guardrail(feedback, audience="customer")
        guardrails["llm"] = llm.allowed
        if not llm.allowed:
            return AgentOutput(
                state={},
                nodes=[],
                guardrails=guardrails,
                simulated_emails=self.simulated_emails - emails_before,
                simulated_tickets=len(self.simulated_tickets) - tickets_before,
            )

        thread_id = f"evaluation:{self.run_id}:{case.id}"
        initial_state = {
            "customer_name": case.input.customer_name,
            "customer_email": case.input.customer_email,
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
        config = {"configurable": {"thread_id": thread_id}}
        executed_nodes: list[str] = []
        known_nodes = {
            "sentiment_analysis",
            "prepare_positive_email",
            "negative_analysis",
            "create_support_ticket",
            "send_internal_notification",
            "prepare_negative_email",
            "human_email_review",
            "revise_email_draft",
            "request_final_email_selection",
            "select_final_email",
            "send_customer_email",
        }

        for update in self.graph.stream(
            initial_state,
            config=config,
            stream_mode="updates",
        ):
            if isinstance(update, dict):
                for node_name in update:
                    if node_name in known_nodes and node_name not in executed_nodes:
                        executed_nodes.append(node_name)

        final_state = dict(self.graph.get_state(config).values)
        predicted_intent = intent_from_state(final_state)
        category = (final_state.get("analysis") or {}).get("issue_category")
        predicted = {
            "sentiment": final_state.get("sentiment"),
            "category": category,
            "intent": predicted_intent,
            "priority": (final_state.get("analysis") or {}).get("priority"),
        }
        final_state["_evaluation_prediction"] = predicted
        return AgentOutput(
            state=final_state,
            nodes=executed_nodes,
            guardrails=guardrails,
            simulated_emails=self.simulated_emails - emails_before,
            simulated_tickets=len(self.simulated_tickets) - tickets_before,
        )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stack.close()
        self._entered = False
