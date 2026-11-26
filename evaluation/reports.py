from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import EvaluationReport


def build_report_payload(
    report: EvaluationReport,
    guardrails: dict[str, Any],
    regression: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "report_format_version": "1",
        "evaluation": report.model_dump(mode="json"),
        "guardrails": guardrails,
        "regression": regression,
    }


def write_json_report(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path
