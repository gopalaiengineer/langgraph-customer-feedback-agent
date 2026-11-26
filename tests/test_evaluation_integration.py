from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from unittest.mock import Mock, patch

import pytest

from app.agents import agent
from app.schemas.schema import EmailDraft, FeedbackAnalysis, SentimentAnalysis
from evaluation.adapter import LangGraphAgentAdapter
from evaluation.data import load_dataset
from evaluation.runner import run_evaluation


class FixedRunnable:
    def __init__(self, value):
        self.value = value

    def invoke(self, _messages):
        return self.value


def test_real_graph_runs_with_memory_and_never_uses_live_side_effects() -> None:
    case = next(
        case
        for case in load_dataset(
            Path(__file__).resolve().parents[1]
            / "evaluation"
            / "datasets"
            / "customer_feedback.json"
        )
        if case.id == "damaged-product"
    )
    analysis = FeedbackAnalysis(
        issue_category="product_quality",
        issue_type="product_quality",
        urgency="high",
        priority="P2",
        recommended_action="Review the damaged item and advise on replacement options.",
        escalation_level="support",
        escalation_reason="The item arrived damaged and cannot be used.",
        tone="frustrated",
        severity="high",
        summary="The item arrived damaged and cannot be used.",
    )
    with (
        patch("app.guardrail.custom_guardrail") as custom,
        patch("app.guardrail.llm_guardrail") as scope,
    ):
        custom.return_value.allowed = True
        scope.return_value.allowed = True
        with LangGraphAgentAdapter(allow_model_calls=True, run_id="offline-test") as adapter:
            with (
                patch.object(
                    adapter._agent,
                    "_get_llm_groq_structred",
                    return_value=FixedRunnable(
                        SentimentAnalysis(sentiment="negative", reason="Problem reported.")
                    ),
                ),
                patch.object(
                    adapter._agent,
                    "_get_llm_gemini_structred",
                    return_value=FixedRunnable(analysis),
                ),
                patch.object(
                    adapter._agent,
                    "_get_llm_email_draft_structured",
                    return_value=FixedRunnable(
                        EmailDraft(subject="We are sorry", body="We are sorry this happened.")
                    ),
                ),
            ):
                report = run_evaluation(
                    [case],
                    adapter,
                    agent_version="test",
                    dataset="synthetic",
                )

        assert report.results[0].execution_status == "completed"
        assert "create_support_ticket" in report.results[0].nodes
        assert "send_internal_notification" in report.results[0].nodes
        assert report.results[0].simulated_tickets == 1
        assert report.results[0].simulated_emails == 1
        assert report.results[0].predicted["category"] == "product_quality"


def test_concurrent_email_attempts_have_one_simulated_delivery() -> None:
    records: dict[str, tuple[str, str | None]] = {}
    lock = Lock()
    deliveries = Mock()

    def claim(key: str):
        with lock:
            if key in records:
                return False, records[key]
            records[key] = ("sending", None)
            return True, None

    def mark_sent(key: str) -> None:
        with lock:
            records[key] = ("sent", None)

    def fake_smtp(*_args, **_kwargs):
        class SMTP:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def ehlo(self):
                return None

            def starttls(self, **_kwargs):
                return None

            def login(self, *_args):
                return None

            def send_message(self, *_args, **_kwargs):
                deliveries()
                return {}

        return SMTP()

    def send() -> None:
        agent._send_email_once("concurrent-test", Mock(), ["fake@example.test"])

    with (
        patch.object(agent._delivery_repository, "claim", side_effect=claim),
        patch.object(agent._delivery_repository, "mark_sent", side_effect=mark_sent),
        patch.object(agent.smtplib, "SMTP", side_effect=fake_smtp),
        ThreadPoolExecutor(max_workers=8) as pool,
    ):
        futures = [pool.submit(send) for _ in range(8)]
        for future in futures:
            try:
                future.result()
            except RuntimeError as error:
                assert "already in progress" in str(error)

    assert deliveries.call_count == 1
    assert records["concurrent-test"] == ("sent", None)


def test_email_failure_is_recorded_and_not_retried_automatically() -> None:
    record: dict[str, tuple[str, str | None]] = {}
    smtp = Mock(side_effect=OSError("simulated SMTP unavailable"))

    def claim(key: str):
        if key in record:
            return False, record[key]
        record[key] = ("sending", None)
        return True, None

    def mark_failed(key: str, error: str) -> None:
        record[key] = ("failed", error)

    with (
        patch.object(agent._delivery_repository, "claim", side_effect=claim),
        patch.object(agent._delivery_repository, "mark_failed", side_effect=mark_failed),
        patch.object(agent.smtplib, "SMTP", smtp),
    ):
        with pytest.raises(OSError, match="simulated SMTP unavailable"):
            agent._send_email_once("failure-test", Mock(), ["fake@example.test"])
        with pytest.raises(RuntimeError, match="will not be retried automatically"):
            agent._send_email_once("failure-test", Mock(), ["fake@example.test"])

    assert record["failure-test"][0] == "failed"
    smtp.assert_called_once()


def test_database_failure_is_not_swallowed() -> None:
    def fail_session():
        raise RuntimeError("isolated database unavailable")

    with patch("app.database.repositories.session_scope", side_effect=fail_session):
        with pytest.raises(RuntimeError, match="isolated database unavailable"):
            agent._ticket_repository.create_or_get(
                "offline-db-test",
                {"ticket_id": "TEST"},
            )
