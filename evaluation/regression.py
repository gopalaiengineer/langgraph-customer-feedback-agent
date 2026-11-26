from __future__ import annotations

from typing import Any

from .models import EvaluationReport


def compare_reports(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
) -> dict[str, Any]:
    if baseline.dataset != candidate.dataset:
        raise ValueError("Regression reports must use the same dataset.")

    baseline_cases = {result.case_id: result for result in baseline.results}
    candidate_cases = {result.case_id: result for result in candidate.results}
    if baseline_cases.keys() != candidate_cases.keys():
        raise ValueError("Regression reports must contain the same case IDs.")
    changed_labels = [
        case_id
        for case_id in baseline_cases
        if baseline_cases[case_id].expected != candidate_cases[case_id].expected
    ]
    if changed_labels:
        raise ValueError(
            "Regression reports must use identical expected labels; changed cases: "
            + ", ".join(sorted(changed_labels))
        )

    metrics = (
        "sentiment_accuracy",
        "intent_accuracy",
        "category_accuracy",
        "action_selection_accuracy",
    )
    changes: dict[str, dict[str, float | None]] = {}
    for metric in metrics:
        before = baseline.summary[metric]["accuracy"]
        after = candidate.summary[metric]["accuracy"]
        changes[metric] = {
            "baseline": before,
            "candidate": after,
            "delta": None if before is None or after is None else after - before,
        }

    regressions = [
        {
            "case_id": case_id,
            "metrics": [
                metric
                for metric in metrics
                if candidate_cases[case_id].deterministic_scores.get(metric) == 0.0
                and baseline_cases[case_id].deterministic_scores.get(metric) == 1.0
            ],
        }
        for case_id in sorted(baseline_cases)
    ]
    regressions = [item for item in regressions if item["metrics"]]
    return {
        "baseline_version": baseline.agent_version,
        "candidate_version": candidate.agent_version,
        "metrics": changes,
        "new_case_regressions": regressions,
    }
