import unittest
from threading import Event
from unittest.mock import Mock, patch

from app.agents import agent
from app.configs.config import settings
from app.helpers import helper
from app.schemas.schema import FeedbackAnalysis
from langgraph.checkpoint.memory import MemorySaver


def analysis(priority: str) -> dict:
    urgency, escalation_level = {
        "P3": ("low", "none"),
        "P2": ("high", "support"),
        "P1": ("critical", "support_and_managers"),
    }[priority]
    return {
        "issue_category": "product_quality",
        "issue_type": "product_quality",
        "urgency": urgency,
        "priority": priority,
        "recommended_action": "Review the reported product issue.",
        "escalation_level": escalation_level,
        "escalation_reason": "The feedback reports a product issue.",
        "tone": "concerned",
        "severity": urgency,
        "summary": "The product did not work as expected.",
    }


class PriorityEscalationTests(unittest.TestCase):
    def setUp(self):
        delivery_records = {}
        ticket_records = {}

        def claim_delivery(idempotency_key):
            if idempotency_key in delivery_records:
                return False, delivery_records[idempotency_key]
            delivery_records[idempotency_key] = ("sending", None)
            return True, None

        def save_ticket(thread_id, ticket):
            return ticket_records.setdefault(thread_id, ticket)

        def mark_delivery_sent(idempotency_key):
            delivery_records[idempotency_key] = ("sent", None)

        def mark_delivery_failed(idempotency_key, error):
            delivery_records[idempotency_key] = ("failed", error)

        for repository, method, side_effect in (
            (agent._delivery_repository, "claim", claim_delivery),
            (agent._delivery_repository, "mark_sent", mark_delivery_sent),
            (agent._delivery_repository, "mark_failed", mark_delivery_failed),
            (agent._ticket_repository, "create_or_get", save_ticket),
        ):
            patcher = patch.object(
                repository,
                method,
                side_effect=side_effect,
            )
            patcher.start()
            self.addCleanup(patcher.stop)

        checkpointer = patch.object(
            agent,
            "get_checkpointer",
            return_value=MemorySaver(),
        )
        checkpointer.start()
        self.addCleanup(checkpointer.stop)

    def test_priority_router_selects_customer_or_escalation_path(self):
        self.assertEqual(
            agent.route_negative_priority({"analysis": analysis("P3")}),
            "prepare_negative_email",
        )
        self.assertEqual(
            agent.route_negative_priority({"analysis": analysis("P2")}),
            ["create_support_ticket", "prepare_negative_email"],
        )
        self.assertEqual(
            agent.route_negative_priority({"analysis": analysis("P1")}),
            ["create_support_ticket", "prepare_negative_email"],
        )

    def test_analysis_rejects_mismatched_priority_escalation(self):
        output = analysis("P1")
        output["escalation_level"] = "support"
        with self.assertRaises(ValueError):
            FeedbackAnalysis(**output)

    def test_analysis_keeps_severity_independent_from_urgency(self):
        output = analysis("P2")
        output["severity"] = "critical"

        parsed = FeedbackAnalysis(**output)

        self.assertEqual(parsed.urgency, "high")
        self.assertEqual(parsed.severity, "critical")

    def test_existing_review_limit_is_configurable_without_changing_flow(self):
        state = {"approval_status": "needs_evaluation", "email_review_attempts": 0}
        with patch.object(settings, "MAX_EMAIL_REVISIONS", 3):
            self.assertEqual(agent.route_email_review(state), "revise")
            state["email_review_attempts"] = 2
            self.assertEqual(
                agent.route_email_review(state),
                "final_selection",
            )

    def test_provider_factories_use_runtime_retry_and_timeout_settings(self):
        with (
            patch.object(settings, "LLM_MAX_RETRIES", 5),
            patch.object(settings, "LLM_REQUEST_TIMEOUT_SECONDS", 42),
            patch.object(settings, "GROQ_API_KEY", "unit-test"),
            patch.object(helper, "ChatGroq") as groq_factory,
            patch.object(settings, "GOOGLE_API_KEY", "unit-test"),
            patch.object(helper, "ChatGoogleGenerativeAI") as google_factory,
        ):
            helper.get_groq_llm()
            helper.get_google_llm()

        self.assertEqual(groq_factory.call_args.kwargs["max_retries"], 5)
        self.assertEqual(groq_factory.call_args.kwargs["timeout"], 42)
        self.assertEqual(google_factory.call_args.kwargs["retries"], 5)
        self.assertEqual(
            google_factory.call_args.kwargs["request_timeout"],
            42,
        )

    def test_ticket_id_is_stable_for_graph_retries(self):
        state = {"analysis": analysis("P2")}
        config = {"configurable": {"thread_id": "ticket-test-thread"}}

        first = agent.create_support_ticket(state, config)["support_ticket"]
        second = agent.create_support_ticket(state, config)["support_ticket"]

        self.assertEqual(first, second)
        self.assertEqual(first["priority"], "P2")

    def test_graph_escalates_negative_feedback_before_human_email_review(self):
        for priority in ("P3", "P2", "P1"):
            with self.subTest(priority=priority):
                initial_state = {
                    "customer_name": "Test Customer",
                    "customer_email": "customer@example.com",
                    "feedback": "The product did not work as expected.",
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
                config = {
                    "configurable": {
                        "thread_id": f"graph-{priority.lower()}-test"
                    }
                }
                with (
                    patch.object(
                        agent,
                        "sentiment_analysis",
                        return_value={"sentiment": "negative"},
                    ),
                    patch.object(
                        agent,
                        "negative_analysis",
                        return_value={"analysis": analysis(priority)},
                    ),
                    patch.object(
                        agent,
                        "prepare_negative_email",
                        return_value={
                            "email_subject": "We apologize",
                            "email_body": "We are sorry to hear this.",
                        },
                    ),
                    patch.object(agent, "_send_email_once"),
                ):
                    graph = agent.agent_manager()
                    graph.invoke(initial_state, config=config)
                    result = graph.get_state(config).values

                self.assertEqual(result["email_status"], "awaiting_approval")
                if priority == "P3":
                    self.assertIsNone(result["support_ticket"])
                    self.assertIsNone(
                        result["internal_notification_status"]
                    )
                else:
                    self.assertEqual(
                        result["support_ticket"]["priority"],
                        priority,
                    )
                    self.assertEqual(
                        result["internal_notification_status"],
                        "sent",
                    )

    def test_customer_email_branch_does_not_wait_for_internal_notification(self):
        initial_state = {
            "customer_name": "Test Customer",
            "customer_email": "customer@example.com",
            "feedback": "The product did not work as expected.",
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
        config = {
            "configurable": {"thread_id": "parallel-branches-test"}
        }
        email_prepared = Event()

        def prepare_email(_state):
            email_prepared.set()
            return {
                "email_subject": "We apologize",
                "email_body": "We are sorry to hear this.",
            }

        def notify_internal(_state):
            self.assertTrue(email_prepared.is_set())
            return {"internal_notification_status": "sent"}

        with (
            patch.object(
                agent,
                "sentiment_analysis",
                return_value={"sentiment": "negative"},
            ),
            patch.object(
                agent,
                "negative_analysis",
                return_value={"analysis": analysis("P2")},
            ),
            patch.object(
                agent,
                "prepare_negative_email",
                side_effect=prepare_email,
            ),
            patch.object(
                agent,
                "send_internal_notification",
                side_effect=notify_internal,
            ),
        ):
            graph = agent.agent_manager()
            graph.invoke(initial_state, config=config)

    def test_positive_branch_still_pauses_for_existing_human_review(self):
        initial_state = {
            "customer_name": "Test Customer",
            "customer_email": "customer@example.com",
            "feedback": "I am happy with the service.",
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
        config = {"configurable": {"thread_id": "positive-branch-test"}}
        with (
            patch.object(
                agent,
                "sentiment_analysis",
                return_value={"sentiment": "positive"},
            ),
            patch.object(
                agent,
                "prepare_positive_email",
                return_value={
                    "email_subject": "Thank you",
                    "email_body": "Thank you for your feedback.",
                },
            ),
        ):
            graph = agent.agent_manager()
            graph.invoke(initial_state, config=config)
            result = graph.get_state(config).values

        self.assertEqual(result["email_status"], "awaiting_approval")
        self.assertEqual(result["email_subject"], "Thank you")
        self.assertIsNone(result["support_ticket"])

    def test_p2_notifies_all_support_recipients_once(self):
        smtp = Mock()
        smtp.__enter__ = Mock(return_value=smtp)
        smtp.__exit__ = Mock(return_value=False)
        smtp.send_message.return_value = {}
        ticket = agent.create_support_ticket(
            {"analysis": analysis("P2")},
            {"configurable": {"thread_id": "p2-test-thread"}},
        )["support_ticket"]
        state = {
            "support_ticket": ticket,
            "internal_notification_status": "pending",
        }

        with patch("app.agents.agent.smtplib.SMTP", return_value=smtp):
            agent.send_internal_notification(
                state,
                {"configurable": {"thread_id": "p2-test-thread"}},
            )
            agent.send_internal_notification(
                state,
                {"configurable": {"thread_id": "p2-test-thread"}},
            )

        smtp.send_message.assert_called_once()
        self.assertEqual(
            smtp.send_message.call_args.kwargs["to_addrs"],
            settings.SUPPORT_EMAILS,
        )
        self.assertNotIn(
            "Cc",
            smtp.send_message.call_args.args[0],
        )

    def test_p1_notifies_support_and_managers_separately_from_customer(self):
        smtp = Mock()
        smtp.__enter__ = Mock(return_value=smtp)
        smtp.__exit__ = Mock(return_value=False)
        smtp.send_message.return_value = {}
        ticket = agent.create_support_ticket(
            {"analysis": analysis("P1")},
            {"configurable": {"thread_id": "p1-test-thread"}},
        )["support_ticket"]
        state = {
            "support_ticket": ticket,
            "internal_notification_status": "pending",
        }

        with patch("app.agents.agent.smtplib.SMTP", return_value=smtp):
            agent.send_internal_notification(
                state,
                {"configurable": {"thread_id": "p1-test-thread"}},
            )

        message = smtp.send_message.call_args.args[0]
        self.assertEqual(
            smtp.send_message.call_args.kwargs["to_addrs"],
            [*settings.SUPPORT_EMAILS, *settings.MANAGER_EMAILS],
        )
        self.assertEqual(message["To"], ", ".join(settings.SUPPORT_EMAILS))
        self.assertEqual(message["Cc"], ", ".join(settings.MANAGER_EMAILS))


if __name__ == "__main__":
    unittest.main()
