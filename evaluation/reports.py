from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .models import EvaluationReport


def write_reports(report: EvaluationReport, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    json_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    columns = [
        "case_id",
        "execution_status",
        "execution_error",
        "prediction_errors",
        "expected",
        "predicted",
        "nodes",
        "guardrails",
        "deterministic_scores",
        "judge_scores",
        "judge_error",
        "simulated_emails",
        "simulated_tickets",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for result in report.results:
            row = result.model_dump()
            for key in ("prediction_errors", "expected", "predicted", "nodes", "guardrails",
                        "deterministic_scores", "judge_scores"):
                row[key] = json.dumps(row[key], ensure_ascii=False)
            writer.writerow(row)
    return json_path, csv_path


def write_guardrail_reports(
    report: dict[str, Any], output_dir: Path, stem: str
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    columns = [
        "case_id",
        "custom_expected",
        "custom_actual",
        "llm_expected",
        "llm_actual",
        "custom_category",
        "llm_category",
        "custom_error",
        "llm_error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(report["cases"])
    return json_path, csv_path
