from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def confusion_metrics(
    expected_allowed: Sequence[bool],
    actual_allowed: Sequence[bool],
) -> dict[str, int | float | None]:
    if len(expected_allowed) != len(actual_allowed):
        raise ValueError("Expected and actual guardrail labels must have equal lengths.")

    tp = sum(expected and actual for expected, actual in zip(expected_allowed, actual_allowed))
    tn = sum(not expected and not actual for expected, actual in zip(expected_allowed, actual_allowed))
    false_acceptance = sum(
        not expected and actual
        for expected, actual in zip(expected_allowed, actual_allowed)
    )
    false_rejection = sum(
        expected and not actual
        for expected, actual in zip(expected_allowed, actual_allowed)
    )
    total = len(expected_allowed)
    return {
        "true_acceptance": tp,
        "true_rejection": tn,
        "false_acceptance": false_acceptance,
        "false_rejection": false_rejection,
        "accuracy": (tp + tn) / total if total else None,
        "evaluated": total,
    }


def evaluate_guardrails(cases: Sequence[dict[str, Any]], *, llm: Any) -> dict[str, Any]:
    from app.guardrail import custom_guardrail, llm_guardrail

    actual_custom: list[bool] = []
    actual_llm: list[bool] = []
    expected_custom: list[bool] = []
    expected_llm: list[bool] = []
    details: list[dict[str, Any]] = []

    for case in cases:
        detail: dict[str, Any] = {
            "case_id": case["id"],
            "custom_expected": case["expected_custom_allowed"],
            "custom_actual": None,
            "llm_expected": case["expected_llm_allowed"],
            "llm_actual": None,
            "custom_category": None,
            "llm_category": None,
            "custom_error": None,
            "llm_error": None,
        }
        try:
            custom = custom_guardrail(case["feedback"])
            actual_custom.append(custom.allowed)
            expected_custom.append(case["expected_custom_allowed"])
            detail["custom_actual"] = custom.allowed
            detail["custom_category"] = custom.category
        except Exception as error:
            detail["custom_error"] = f"{type(error).__name__}: {error}"

        try:
            scope = llm_guardrail(case["feedback"], llm=llm)
            actual_llm.append(scope.allowed)
            expected_llm.append(case["expected_llm_allowed"])
            detail["llm_actual"] = scope.allowed
            detail["llm_category"] = scope.category
        except Exception as error:
            detail["llm_error"] = f"{type(error).__name__}: {error}"
        details.append(detail)

    return {
        "custom_guardrail": confusion_metrics(expected_custom, actual_custom),
        "llm_guardrail": confusion_metrics(expected_llm, actual_llm),
        "failed_custom_checks": sum(item["custom_error"] is not None for item in details),
        "failed_llm_checks": sum(item["llm_error"] is not None for item in details),
        "cases": details,
    }
