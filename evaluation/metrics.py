from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .models import CaseResult, EvaluationCase


def accuracy(pairs: Sequence[tuple[Any, Any]]) -> float | None:
    if not pairs:
        return None
    return sum(expected == predicted for expected, predicted in pairs) / len(pairs)


def intent_from_state(state: dict[str, Any]) -> str | None:
    sentiment = state.get("sentiment")
    if sentiment == "positive":
        return "positive_feedback"
    analysis = state.get("analysis") or {}
    category = analysis.get("issue_category")
    if not isinstance(category, str):
        return None
    if category == "refund":
        return "refund_request"
    return f"{category}_complaint"


def selected_actions(nodes: Sequence[str]) -> list[str]:
    operational_nodes = {
        "prepare_positive_email",
        "prepare_negative_email",
        "create_support_ticket",
        "send_internal_notification",
    }
    return sorted(operational_nodes.intersection(nodes))


def score_case(
    case: EvaluationCase,
    predicted: dict[str, Any],
    nodes: Sequence[str] = (),
) -> dict[str, float | None]:
    expected = case.expected
    expected_actions = set(expected.actions)
    observed_actions = set(selected_actions(nodes))
    if expected.sentiment == "positive":
        expected_route_observed = (
            "prepare_positive_email" in nodes and "negative_analysis" not in nodes
        )
    else:
        expected_route_observed = "negative_analysis" in nodes
        if expected.priority == "P3":
            expected_route_observed &= "create_support_ticket" not in nodes
        elif expected.priority in {"P1", "P2"}:
            expected_route_observed &= (
                "create_support_ticket" in nodes
                and "send_internal_notification" in nodes
            )
    return {
        "sentiment_accuracy": float(predicted.get("sentiment") == expected.sentiment),
        "intent_accuracy": float(predicted.get("intent") == expected.intent),
        "category_accuracy": (
            None
            if expected.category is None
            else float(predicted.get("category") == expected.category)
        ),
        "action_selection_accuracy": float(
            set(predicted.get("actions", [])) == expected_actions
        ),
        "conditional_route_accuracy": float(expected_route_observed),
        "node_execution_accuracy": float(observed_actions == expected_actions),
        # This graph has no LangChain Tool nodes; these operational nodes are
        # the closest existing observable equivalents.
        "tool_selection_accuracy": float(observed_actions == expected_actions),
        "tool_execution_accuracy": float(observed_actions == expected_actions),
    }


def aggregate_scores(results: Sequence[CaseResult]) -> dict[str, Any]:
    score_names = (
        "sentiment_accuracy",
        "intent_accuracy",
        "category_accuracy",
        "action_selection_accuracy",
        "conditional_route_accuracy",
        "node_execution_accuracy",
        "tool_selection_accuracy",
        "tool_execution_accuracy",
    )
    aggregate: dict[str, Any] = {}
    for name in score_names:
        values = [
            result.deterministic_scores[name]
            for result in results
            if result.execution_status == "completed"
            and result.deterministic_scores.get(name) is not None
        ]
        aggregate[name] = {
            "accuracy": sum(values) / len(values) if values else None,
            "evaluated": len(values),
        }
    aggregate["executions"] = {
        "total": len(results),
        "completed": sum(result.execution_status == "completed" for result in results),
        "guardrail_rejected": sum(
            result.execution_status == "guardrail_rejected" for result in results
        ),
        "failed": sum(result.execution_status == "failed" for result in results),
        "prediction_error_cases": sum(bool(result.prediction_errors) for result in results),
        "tool_nodes_available": 0,
        "tool_node_note": (
            "The application has no LangChain tool nodes; tool metrics use the "
            "existing support-ticket and internal-notification graph nodes."
        ),
    }
    judge_dimensions = ("relevance", "helpfulness", "empathy", "accuracy", "clarity")
    aggregate["llm_judge"] = {
        dimension: {
            "mean": (
                sum(
                    result.judge_scores[dimension]
                    for result in results
                    if result.judge_scores is not None
                )
                / sum(result.judge_scores is not None for result in results)
                if any(result.judge_scores is not None for result in results)
                else None
            ),
            "evaluated": sum(result.judge_scores is not None for result in results),
        }
        for dimension in judge_dimensions
    }
    aggregate["llm_judge"]["failed_cases"] = sum(
        result.judge_error is not None for result in results
    )
    return aggregate
