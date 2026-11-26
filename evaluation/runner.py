from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Protocol, Any

from .metrics import aggregate_scores, score_case, selected_actions
from .models import AgentOutput, CaseResult, EvaluationCase, EvaluationReport


class AgentRunner(Protocol):
    def invoke(self, case: EvaluationCase) -> AgentOutput: ...


Judge = Callable[[EvaluationCase, str], dict[str, int]]


def run_evaluation(
    cases: Sequence[EvaluationCase],
    agent: AgentRunner,
    *,
    agent_version: str,
    dataset: str,
    judge: Judge | None = None,
) -> EvaluationReport:
    started = datetime.now(timezone.utc)
    results: list[CaseResult] = []

    for case in cases:
        expected = case.expected.model_dump()
        try:
            output = agent.invoke(case)
            if not all(output.guardrails.values()):
                results.append(
                    CaseResult(
                        case_id=case.id,
                        execution_status="guardrail_rejected",
                        expected=expected,
                        guardrails=output.guardrails,
                    )
                )
                continue

            predicted = output.state.get("_evaluation_prediction", {})
            predicted["actions"] = selected_actions(output.nodes)
            score_values = score_case(case, predicted, output.nodes)
            errors = [
                name
                for name, value in score_values.items()
                if value == 0.0
            ]
            result = CaseResult(
                case_id=case.id,
                execution_status="completed",
                expected=expected,
                predicted=predicted,
                nodes=output.nodes,
                guardrails=output.guardrails,
                deterministic_scores=score_values,
                prediction_errors=errors,
                simulated_emails=output.simulated_emails,
                simulated_tickets=output.simulated_tickets,
            )
            if judge is not None:
                response = "\n".join(
                    part
                    for part in (
                        output.state.get("email_subject") or "",
                        output.state.get("email_body") or "",
                    )
                    if part
                )
                try:
                    result.judge_scores = judge(case, response)
                except Exception as error:
                    result.judge_error = f"{type(error).__name__}: {error}"
            results.append(result)
        except Exception as error:
            results.append(
                CaseResult(
                    case_id=case.id,
                    execution_status="failed",
                    execution_error=f"{type(error).__name__}: {error}",
                    expected=expected,
                )
            )

    finished = datetime.now(timezone.utc)
    return EvaluationReport(
        agent_version=agent_version,
        dataset=dataset,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        summary=aggregate_scores(results),
        results=results,
    )
