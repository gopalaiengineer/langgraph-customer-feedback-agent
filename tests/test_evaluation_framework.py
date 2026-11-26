from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from evaluation.data import load_dataset, load_guardrail_dataset
from evaluation.guardrails import confusion_metrics, evaluate_guardrails
from evaluation.metrics import score_case
from evaluation.models import AgentOutput, EvaluationCase
from evaluation.regression import compare_reports
from evaluation.reports import write_reports
from evaluation.runner import run_evaluation


DATASET = Path(__file__).resolve().parents[1] / "evaluation" / "datasets"


class FakeAgent:
    def __init__(self, outputs: dict[str, AgentOutput | Exception]) -> None:
        self.outputs = outputs

    def invoke(self, case: EvaluationCase) -> AgentOutput:
        output = self.outputs[case.id]
        if isinstance(output, Exception):
            raise output
        return output


def test_dataset_uses_unique_cases_and_real_graph_outcomes() -> None:
    cases = load_dataset(DATASET / "customer_feedback.json")
    assert len(cases) >= 6
    assert len({case.id for case in cases}) == len(cases)
    assert any(case.expected.category is None for case in cases)
    assert any("create_support_ticket" in case.expected.actions for case in cases)


def test_guardrail_false_acceptance_and_rejection_are_separate() -> None:
    result = confusion_metrics(
        [True, True, False, False],
        [True, False, True, False],
    )
    assert result["false_acceptance"] == 1
    assert result["false_rejection"] == 1
    assert result["accuracy"] == 0.5
    assert result["evaluated"] == 4


def test_guardrail_dataset_contains_safe_and_rejected_inputs() -> None:
    cases = load_guardrail_dataset(DATASET / "guardrails.json")
    assert any(case["expected_llm_allowed"] for case in cases)
    assert any(not case["expected_llm_allowed"] for case in cases)


def test_guardrail_evaluation_records_errors_and_continues() -> None:
    cases = [
        {
            "id": "broken-custom",
            "feedback": "Synthetic feedback",
            "expected_custom_allowed": True,
            "expected_llm_allowed": True,
        },
        {
            "id": "next-case",
            "feedback": "More synthetic feedback",
            "expected_custom_allowed": True,
            "expected_llm_allowed": True,
        },
    ]
    with (
        patch(
            "app.guardrail.custom_guardrail",
            side_effect=[
                RuntimeError("synthetic custom failure"),
                SimpleNamespace(allowed=True, category="customer_feedback"),
            ],
        ),
        patch(
            "app.guardrail.llm_guardrail",
            side_effect=[
                SimpleNamespace(allowed=True, category="customer_feedback"),
                SimpleNamespace(allowed=True, category="customer_feedback"),
            ],
        ),
    ):
        report = evaluate_guardrails(cases, llm=object())

    assert report["failed_custom_checks"] == 1
    assert report["custom_guardrail"]["evaluated"] == 1
    assert report["cases"][0]["custom_error"] == "RuntimeError: synthetic custom failure"


def test_case_scoring_skips_unavailable_positive_category() -> None:
    case = next(
        case
        for case in load_dataset(DATASET / "customer_feedback.json")
        if case.expected.category is None
    )
    scores = score_case(
        case,
        {
            "sentiment": "positive",
            "intent": "positive_feedback",
            "category": None,
            "actions": case.expected.actions,
        },
    )
    assert scores["sentiment_accuracy"] == 1.0
    assert scores["category_accuracy"] is None


def test_runner_keeps_execution_failures_out_of_prediction_scores() -> None:
    cases = load_dataset(DATASET / "customer_feedback.json")[:2]
    outputs = {
        cases[0].id: AgentOutput(
            state={
                "_evaluation_prediction": {
                    "sentiment": cases[0].expected.sentiment,
                    "category": None,
                    "intent": cases[0].expected.intent,
                }
            },
            nodes=cases[0].expected.actions,
            guardrails={"custom": True, "llm": True},
        ),
        cases[1].id: RuntimeError("synthetic execution failure"),
    }
    report = run_evaluation(
        cases,
        FakeAgent(outputs),
        agent_version="test",
        dataset="test-dataset",
    )
    assert report.summary["executions"]["completed"] == 1
    assert report.summary["executions"]["failed"] == 1
    assert report.results[1].execution_error == "RuntimeError: synthetic execution failure"
    assert report.results[1].deterministic_scores == {}


def test_regression_detects_new_per_case_errors() -> None:
    case = load_dataset(DATASET / "customer_feedback.json")[0]
    correct = {
        "_evaluation_prediction": {
            "sentiment": case.expected.sentiment,
            "category": None,
            "intent": case.expected.intent,
        }
    }
    baseline = run_evaluation(
        [case],
        FakeAgent({
            case.id: AgentOutput(
                state=correct,
                nodes=case.expected.actions,
                guardrails={"custom": True, "llm": True},
            )
        }),
        agent_version="v1",
        dataset="same",
    )
    candidate = run_evaluation(
        [case],
        FakeAgent({
            case.id: AgentOutput(
                state={
                    "_evaluation_prediction": {
                        "sentiment": "negative",
                        "category": None,
                        "intent": "positive_feedback",
                    }
                },
                nodes=case.expected.actions,
                guardrails={"custom": True, "llm": True},
            )
        }),
        agent_version="v2",
        dataset="same",
    )
    comparison = compare_reports(baseline, candidate)
    assert comparison["metrics"]["sentiment_accuracy"]["delta"] == -1.0
    assert comparison["new_case_regressions"][0]["case_id"] == case.id


def test_model_call_opt_in_is_required() -> None:
    from evaluation.adapter import LangGraphAgentAdapter

    with pytest.raises(ValueError, match="external LLM providers"):
        LangGraphAgentAdapter(allow_model_calls=False, run_id="test")


def test_reports_include_detailed_json_and_csv(tmp_path: Path) -> None:
    case = load_dataset(DATASET / "customer_feedback.json")[0]
    output = AgentOutput(
        state={
            "_evaluation_prediction": {
                "sentiment": case.expected.sentiment,
                "category": None,
                "intent": case.expected.intent,
            }
        },
        nodes=case.expected.actions,
        guardrails={"custom": True, "llm": True},
    )
    report = run_evaluation(
        [case],
        FakeAgent({case.id: output}),
        agent_version="test",
        dataset="synthetic",
    )
    json_path, csv_path = write_reports(report, tmp_path, "sample")
    assert '"execution_status": "completed"' in json_path.read_text(encoding="utf-8")
    assert "case_id,execution_status,execution_error" in csv_path.read_text(
        encoding="utf-8"
    )
