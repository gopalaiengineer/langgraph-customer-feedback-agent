from contextlib import contextmanager
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

from sqlalchemy.dialects import postgresql

from app.agents import agent
from app.database.repositories import SupportTicketRepository
from app.pipelines import pipeline
from app.schemas.schema import EmailDraft


class TicketResolutionTests(unittest.TestCase):
    def setUp(self):
        self.ticket = {
            "thread_id": "thread-123",
            "ticket_id": "FB-123456789ABC",
            "ticket_data": {
                "status": "open",
                "customer_name": "Alex Customer",
                "customer_email": "alex@example.com",
                "summary": "The replacement order was delayed.",
            },
        }

    def test_build_resolution_email_uses_ticket_and_admin_details(self):
        resolution = "We shipped a replacement and confirmed the new delivery date."
        generated_draft = EmailDraft(
            subject="Update on your replacement order",
            body=(
                "Dear Alex Customer,\n"
                "We reviewed the delayed replacement order. "
                f"{resolution}"
            ),
        )
        with patch.object(
            pipeline,
            "draft_ticket_resolution_email",
            return_value=generated_draft,
        ) as draft_email:
            recipient, subject, body = pipeline.build_ticket_resolution_email(
                self.ticket["ticket_id"],
                self.ticket["ticket_data"],
                resolution,
            )

        self.assertEqual(recipient, "alex@example.com")
        self.assertEqual(subject, generated_draft.subject)
        self.assertIn("Dear Alex Customer", body)
        self.assertIn(resolution, body)
        draft_email.assert_called_once_with(
            self.ticket["ticket_id"],
            "Alex Customer",
            self.ticket["ticket_data"]["summary"],
            resolution,
        )

    def test_build_resolution_email_rejects_missing_or_invalid_recipient(self):
        for recipient in ("", "not-an-email"):
            with self.subTest(recipient=recipient):
                data = {**self.ticket["ticket_data"], "customer_email": recipient}
                with self.assertRaisesRegex(ValueError, "registered customer"):
                    pipeline.build_ticket_resolution_email(
                        self.ticket["ticket_id"],
                        data,
                        "The issue has been addressed.",
                    )

    def test_resolve_ticket_sends_before_persisting_resolved_status(self):
        pending = {
            "reason": "The issue has been addressed.",
            "email_subject": "Update on your support ticket",
            "email_body": "Resolution details",
        }
        with (
            patch.object(
                pipeline._ticket_repository,
                "get_by_ticket_id",
                return_value=self.ticket,
            ),
            patch.object(
                pipeline,
                "build_ticket_resolution_email",
                return_value=(
                    "alex@example.com",
                    "Update on ticket",
                    "LLM-generated email",
                ),
            ),
            patch.object(
                pipeline._ticket_repository,
                "begin_resolution",
                return_value=pending,
            ) as begin_resolution,
            patch.object(pipeline, "send_ticket_resolution_email") as send_email,
            patch.object(
                pipeline._ticket_repository,
                "mark_resolved",
                return_value={"status": "resolved"},
            ) as mark_resolved,
        ):
            result = pipeline.resolve_support_ticket(
                self.ticket["ticket_id"],
                "  The issue has been addressed.  ",
                "support-admin",
            )

        self.assertEqual(result["status"], "resolved")
        begin_resolution.assert_called_once_with(
            "thread-123",
            "The issue has been addressed.",
            ANY,
            ANY,
        )
        send_email.assert_called_once_with(
            "thread-123",
            self.ticket["ticket_id"],
            "alex@example.com",
            pending["email_subject"],
            pending["email_body"],
        )
        mark_resolved.assert_called_once_with(
            "thread-123",
            pending["reason"],
            "support-admin",
            pending["email_subject"],
            pending["email_body"],
        )

    def test_retry_reuses_pending_llm_email_without_generating_another(self):
        pending = {
            "reason": "The issue has been addressed.",
            "email_subject": "Existing generated subject",
            "email_body": "Existing generated body",
        }
        ticket = {
            **self.ticket,
            "ticket_data": {
                **self.ticket["ticket_data"],
                "resolution_pending": pending,
            },
        }
        with (
            patch.object(
                pipeline._ticket_repository,
                "get_by_ticket_id",
                return_value=ticket,
            ),
            patch.object(
                pipeline,
                "build_ticket_resolution_email",
            ) as build_email,
            patch.object(
                pipeline._ticket_repository,
                "begin_resolution",
                return_value=pending,
            ),
            patch.object(pipeline, "send_ticket_resolution_email") as send_email,
            patch.object(
                pipeline._ticket_repository,
                "mark_resolved",
                return_value={"status": "resolved"},
            ),
        ):
            result = pipeline.resolve_support_ticket(
                self.ticket["ticket_id"],
                pending["reason"],
                "support-admin",
            )

        self.assertEqual(result["status"], "resolved")
        build_email.assert_not_called()
        send_email.assert_called_once_with(
            "thread-123",
            self.ticket["ticket_id"],
            "alex@example.com",
            pending["email_subject"],
            pending["email_body"],
        )

    def test_reject_ticket_updates_status_without_sending_email(self):
        rejected = {"status": "rejected"}
        with (
            patch.object(
                pipeline._ticket_repository,
                "get_by_ticket_id",
                return_value=self.ticket,
            ),
            patch.object(
                pipeline._ticket_repository,
                "mark_rejected",
                return_value=rejected,
            ) as mark_rejected,
            patch.object(pipeline, "send_ticket_resolution_email") as send_email,
        ):
            result = pipeline.reject_support_ticket(self.ticket["ticket_id"])

        self.assertEqual(result["status"], "rejected")
        mark_rejected.assert_called_once_with("thread-123")
        send_email.assert_not_called()

    def test_new_ticket_saves_customer_contact_for_resolution_workflow(self):
        state = {
            "customer_name": "Alex Customer",
            "customer_email": "alex@example.com",
            "analysis": {
                "priority": "P2",
                "issue_category": "billing",
                "summary": "A duplicate charge appeared.",
                "recommended_action": "Review the duplicate charge.",
                "escalation_level": "support",
                "escalation_reason": "The charge remains unresolved.",
            },
        }
        with patch.object(
            agent._ticket_repository,
            "create_or_get",
            side_effect=lambda thread_id, ticket: ticket,
        ) as create_or_get:
            agent.create_support_ticket(
                state,
                {"configurable": {"thread_id": "thread-123"}},
            )

        saved_ticket = create_or_get.call_args.args[1]
        self.assertEqual(saved_ticket["customer_name"], "Alex Customer")
        self.assertEqual(saved_ticket["customer_email"], "alex@example.com")

    def test_begin_resolution_locks_ticket_and_preserves_a_single_pending_email(self):
        record = SimpleNamespace(
            thread_id="thread-123",
            ticket_data={"status": "open"},
        )
        session = Mock()
        session.scalar.return_value = record

        @contextmanager
        def fake_session_scope():
            yield session

        with patch(
            "app.database.repositories.session_scope",
            side_effect=fake_session_scope,
        ):
            pending = SupportTicketRepository().begin_resolution(
                "thread-123",
                "Clarification",
                "Subject",
                "Body",
            )

        self.assertEqual(pending["reason"], "Clarification")
        self.assertEqual(record.ticket_data["resolution_pending"], pending)
        statement = session.scalar.call_args.args[0]
        self.assertIn(
            "FOR UPDATE",
            str(statement.compile(dialect=postgresql.dialect())),
        )

    def test_mark_resolved_appends_history_and_clears_pending_resolution(self):
        pending = {
            "reason": "Clarification",
            "email_subject": "Subject",
            "email_body": "Body",
        }
        record = SimpleNamespace(
            thread_id="thread-123",
            ticket_data={
                "status": "open",
                "resolution_pending": pending,
            },
        )
        session = Mock()
        session.scalar.return_value = record

        @contextmanager
        def fake_session_scope():
            yield session

        with patch(
            "app.database.repositories.session_scope",
            side_effect=fake_session_scope,
        ):
            resolved = SupportTicketRepository().mark_resolved(
                "thread-123",
                "Clarification",
                "support-admin",
                "Subject",
                "Body",
            )

        self.assertEqual(resolved["status"], "resolved")
        self.assertNotIn("resolution_pending", resolved)
        self.assertEqual(len(resolved["resolution_history"]), 1)
        self.assertEqual(
            resolved["resolution_history"][0]["resolved_by"],
            "support-admin",
        )

    def test_mark_rejected_only_changes_status_and_preserves_ticket_data(self):
        original_data = {
            "status": "open",
            "summary": "The issue could not be verified.",
            "priority": "P2",
        }
        record = SimpleNamespace(
            thread_id="thread-123",
            ticket_data=dict(original_data),
        )
        session = Mock()
        session.scalar.return_value = record

        @contextmanager
        def fake_session_scope():
            yield session

        with patch(
            "app.database.repositories.session_scope",
            side_effect=fake_session_scope,
        ):
            rejected = SupportTicketRepository().mark_rejected("thread-123")

        self.assertEqual(rejected, {**original_data, "status": "rejected"})


if __name__ == "__main__":
    unittest.main()
