from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

from .data import load_dataset, load_guardrail_dataset
from .models import EvaluationReport
from .reports import write_guardrail_reports, write_reports
from .runner import run_evaluation


ROOT = Path(__file__).resolve().parents[1]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the LangGraph agent with isolated email/database side effects."
    )
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation" / "datasets" / "customer_feedback.json")
    parser.add_argument("--guardrail-dataset", type=Path, default=ROOT / "evaluation" / "datasets" / "guardrails.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "reports")
    parser.add_argument("--version", default=os.environ.get("AGENT_VERSION", "working-tree"))
    parser.add_argument(
        "--allow-model-calls",
        action="store_true",
        help="Permit synthetic evaluation text to be sent to configured LLM providers.",
    )
    parser.add_argument("--judge", action="store_true", help="Also request LLM-as-a-Judge scores.")
    parser.add_argument("--baseline", type=Path, help="Compare against a prior JSON evaluation report.")
    return parser.parse_args()


def _make_model() -> Any:
    from app.configs.config import settings
    from app.helpers.helper import get_llm

    provider = (
        "groq"
        if settings.ENABLE_GROQ
        else "gemini"
        if settings.ENABLE_GEMINI
        else "openai"
    )
    return get_llm(provider=provider, temperature=0)


def main() -> int:
    args = _arguments()
    if not args.allow_model_calls:
        raise SystemExit(
            "Refusing to call the real agent without explicit opt-in. "
            "Add --allow-model-calls; all evaluation prompts are synthetic."
        )

    cases = load_dataset(args.dataset)
    guardrail_cases = load_guardrail_dataset(args.guardrail_dataset)
    run_id = uuid.uuid4().hex[:12]

    from .adapter import LangGraphAgentAdapter
    from .guardrails import evaluate_guardrails
    from .judge import configured_judge

    with LangGraphAgentAdapter(allow_model_calls=True, run_id=run_id) as adapter:
        report = run_evaluation(
            cases,
            adapter,
            agent_version=args.version,
            dataset=str(args.dataset.resolve()),
            judge=configured_judge() if args.judge else None,
        )
        guardrail_report = evaluate_guardrails(
            guardrail_cases,
            llm=_make_model(),
        )

    json_path, csv_path = write_reports(report, args.output, f"evaluation-{run_id}")
    guardrail_json, guardrail_csv = write_guardrail_reports(
        guardrail_report,
        args.output,
        f"guardrails-{run_id}",
    )

    print(json.dumps(report.summary, indent=2))
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")
    print(f"Guardrail JSON report: {guardrail_json}")
    print(f"Guardrail CSV report: {guardrail_csv}")

    if args.baseline:
        from .regression import compare_reports

        baseline = EvaluationReport.model_validate_json(
            args.baseline.read_text(encoding="utf-8")
        )
        comparison = compare_reports(baseline, report)
        regression_path = args.output / f"regression-{run_id}.json"
        regression_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"Regression report: {regression_path}")
    return 0
