import re
import unicodedata
from collections.abc import Sequence

from app.guardrail.schemas import CustomGuardrailResult

ALLOWED_KEYWORDS = [
    # Customer feedback
    "customer feedback",
    "customer complaint",
    "customer experience",
    "customer satisfaction",
    "customer review",
    "customer response",
    "customer issue",
    "customer concern",
    "customer suggestion",
    # Sentiment analysis
    "sentiment",
    "sentiment analysis",
    "positive feedback",
    "negative feedback",
    "neutral feedback",
    "feedback analysis",
    # Product and service issues
    "product quality",
    "product defect",
    "damaged product",
    "damaged",
    "delivery",
    "delivery issue",
    "shipping issue",
    "billing issue",
    "payment issue",
    "refund",
    "cancellation",
    "technical support",
    "customer service",
    # Priority and escalation
    "priority",
    "p1",
    "p2",
    "p3",
    "urgency",
    "severity",
    "escalation",
    "manager escalation",
    "support escalation",
    # Ticket management
    "support ticket",
    "ticket creation",
    "ticket status",
    "ticket management",
    # Email workflow
    "customer email",
    "apology email",
    "email approval",
    "email review",
    "email revision",
    "send email",
    # Project-related workflow
    "human in the loop",
    "approval workflow",
    "feedback workflow",
    "customer feedback management",
    "langgraph feedback agent",
]

RESTRICTED_KEYWORDS = [
    # Prompt injection
    "ignore previous instructions",
    "ignore all instructions",
    "forget your instructions",
    "override system prompt",
    "bypass system prompt",
    "act as unrestricted",
    "jailbreak",
    # Internal information
    "reveal system prompt",
    "show system prompt",
    "hidden instructions",
    "internal instructions",
    "reveal secrets",
    "show api key",
    "api key",
    "show password",
    "database credentials",
    "smtp password",
    "environment variables",
    # Unauthorized actions
    "bypass approval",
    "disable approval",
    "skip human approval",
    "send without approval",
    "access another customer",
    "show other customer data",
    "delete all customer records",
    "dump database",
]

REVIEW_ALLOWED_KEYWORDS = [
    "email tone",
    "tone",
    "wording",
    "rewrite",
    "revise",
    "shorter",
    "more concise",
    "more professional",
    "clarify",
]

MIN_INPUT_LENGTH = 1
MAX_INPUT_LENGTH = 5000

_QUOTE_PAIRS = {
    '"': '"',
    "'": "'",
    "“": "”",
    "‘": "’",
    "«": "»",
}


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return " ".join(normalized.split())


def _mask_quoted_content(text: str) -> str:
    """Mask quoted text before restricted matching; quoted customer content is untrusted."""
    characters = list(text)
    index = 0
    while index < len(text):
        quote = text[index]
        closing_quote = _QUOTE_PAIRS.get(quote)
        if closing_quote is None:
            index += 1
            continue

        if (
            quote == "'"
            and index > 0
            and index + 1 < len(text)
            and text[index - 1].isalnum()
            and text[index + 1].isalnum()
        ):
            index += 1
            continue

        closing_index = text.find(closing_quote, index + 1)
        if closing_index == -1:
            index += 1
            continue

        for masked_index in range(index, closing_index + 1):
            characters[masked_index] = " "
        index = closing_index + 1

    return "".join(characters)


def _find_matches(text: str, keywords: Sequence[str]) -> list[str]:
    matches = []
    for keyword in keywords:
        normalized_keyword = _normalize(keyword)
        if normalized_keyword and re.search(
            rf"(?<!\w){re.escape(normalized_keyword)}(?!\w)",
            text,
        ):
            matches.append(keyword)
    return matches


def custom_guardrail(
    user_input: str,
    *,
    allowed_keywords: Sequence[str] | None = None,
    restricted_keywords: Sequence[str] | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
) -> CustomGuardrailResult:
    """Perform deterministic preliminary validation without calling an LLM."""
    if not isinstance(user_input, str):
        return CustomGuardrailResult(
            allowed=False,
            reason="Input must be a string.",
            matched_keywords=[],
            restricted_matches=[],
            category="invalid_input",
        )

    if not user_input:
        return CustomGuardrailResult(
            allowed=False,
            reason="Input must not be empty.",
            matched_keywords=[],
            restricted_matches=[],
            category="invalid_input",
        )

    if not user_input.strip():
        return CustomGuardrailResult(
            allowed=False,
            reason="Input must not contain only whitespace.",
            matched_keywords=[],
            restricted_matches=[],
            category="invalid_input",
        )

    effective_min_length = MIN_INPUT_LENGTH if min_length is None else min_length
    effective_max_length = MAX_INPUT_LENGTH if max_length is None else max_length
    if (
        effective_min_length < 0
        or effective_max_length < effective_min_length
    ):
        raise ValueError("Input length limits must be non-negative and ordered.")

    if len(user_input) < effective_min_length:
        return CustomGuardrailResult(
            allowed=False,
            reason=f"Input must be at least {effective_min_length} characters long.",
            matched_keywords=[],
            restricted_matches=[],
            category="invalid_input",
        )

    if len(user_input) > effective_max_length:
        return CustomGuardrailResult(
            allowed=False,
            reason=f"Input must not exceed {effective_max_length} characters.",
            matched_keywords=[],
            restricted_matches=[],
            category="invalid_input",
        )

    normalized_input = _normalize(user_input)
    normalized_unquoted_input = _mask_quoted_content(normalized_input)
    effective_restricted_keywords = (
        RESTRICTED_KEYWORDS
        if restricted_keywords is None
        else restricted_keywords
    )
    restricted_matches = _find_matches(
        normalized_unquoted_input,
        effective_restricted_keywords,
    )
    if restricted_matches:
        return CustomGuardrailResult(
            allowed=False,
            reason="Restricted instruction or information-access pattern detected.",
            matched_keywords=[],
            restricted_matches=restricted_matches,
            category="restricted_request",
        )

    effective_allowed_keywords = (
        ALLOWED_KEYWORDS if allowed_keywords is None else allowed_keywords
    )
    matched_keywords = _find_matches(normalized_input, effective_allowed_keywords)
    if not matched_keywords:
        return CustomGuardrailResult(
            allowed=False,
            reason="No relevant project keywords detected.",
            matched_keywords=[],
            restricted_matches=[],
            category="out_of_scope",
        )

    return CustomGuardrailResult(
        allowed=True,
        reason=(
            "Relevant project keywords detected; this preliminary check does "
            "not confirm the request is safe or in scope."
        ),
        matched_keywords=matched_keywords,
        restricted_matches=[],
        category="customer_feedback",
    )
