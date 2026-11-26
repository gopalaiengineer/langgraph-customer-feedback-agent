from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import EvaluationCase


def load_dataset(path: Path) -> list[EvaluationCase]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Dataset must contain a JSON array: {path}")
    cases = [EvaluationCase.model_validate(item) for item in raw]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Dataset case IDs must be unique: {path}")
    return cases


def load_guardrail_dataset(path: Path) -> list[dict[str, Any]]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Guardrail dataset must contain a JSON array: {path}")
    required = {
        "id",
        "feedback",
        "expected_custom_allowed",
        "expected_llm_allowed",
    }
    if any(not isinstance(item, dict) or set(item) != required for item in raw):
        raise ValueError("Guardrail cases must use the documented exact fields.")
    ids = [item["id"] for item in raw]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Guardrail case IDs must be unique: {path}")
    for item in raw:
        if not isinstance(item["feedback"], str) or not item["feedback"]:
            raise ValueError(f"Guardrail case {item['id']} needs non-empty feedback.")
        if not isinstance(item["expected_custom_allowed"], bool) or not isinstance(
            item["expected_llm_allowed"], bool
        ):
            raise ValueError(f"Guardrail case {item['id']} needs boolean labels.")
    return raw
