from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

from .data import load_dataset, load_guardrail_dataset
from .models import EvaluationReport
from .pdf_report import write_pdf_report
from .reports import build_report_payload, write_json_report
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

    regression = None
    if args.baseline:
        baseline_data = json.loads(args.baseline.read_text(encoding="utf-8"))
        baseline_payload = baseline_data.get("evaluation", baseline_data)
        baseline = EvaluationReport.model_validate(baseline_payload)
        from .regression import compare_reports

        regression = compare_reports(baseline, report)

    stem = f"evaluation-{run_id}"
    json_payload = build_report_payload(report, guardrail_report, regression)
    json_path = write_json_report(json_payload, args.output / f"{stem}.json")
    pdf_path = write_pdf_report(
        report,
        guardrail_report,
        args.output / f"{stem}.pdf",
        regression=regression,
    )

    print(json.dumps(report.summary, indent=2))
    print(f"JSON report: {json_path}")
    print(f"PDF report: {pdf_path}")
    return 0
