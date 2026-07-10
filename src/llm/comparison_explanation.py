# ===============================
# src/llm/comparison_explanation.py
# ===============================
#
# Phase 15.5 — AI Property Comparison explanation.
#
# Thin orchestration layer that turns an EXISTING backend comparison result
# (the structured { "winner": {...}, "rankings": [...] } produced by the
# comparison agent) into a natural-language explanation of the comparison.
#
# It performs NO business logic:
#   - It does not compare properties, score them, or rank them.
#   - It never overrides the winner, re-ranks the options, or invents scores,
#     prices, amenities or comparison points.
#   - It only reuses the Phase 15.2 Comparison Prompt Builder + the Phase 15.1
#     Claude Client to explain the structured result the backend already
#     produced.
#
# Claude is OPTIONAL. If the explanation cannot be generated for any reason,
# this module returns None and the caller still returns the backend comparison
# unchanged.

import logging

from src.llm.claude_client import ask_claude
from src.llm.prompts import build_comparison_prompt

logger = logging.getLogger(__name__)


def explain_comparison(comparison: dict) -> str | None:
    """
    Generate a conversational explanation for a backend comparison result.

    Args:
        comparison : The structured comparison response already produced by the
                     backend (compare_properties), containing a "winner" record
                     and a "rankings" list. The backend has already scored and
                     ranked the properties; this only explains that result.

    Returns:
        The explanation text, or None when Claude is unavailable / fails or
        there is nothing to explain. Returning None lets the caller keep the
        backend comparison working without an explanation.
    """

    # Nothing to explain — no result, an error payload, or no rankings.
    # Do not spend a Claude call and never try to explain a failed comparison.
    if not comparison or comparison.get("error") or not comparison.get("rankings"):
        return None

    # Reuse the Phase 15.2 builder; never build prompts inline here.
    prompt = build_comparison_prompt(comparison)

    # Reuse the Phase 15.1 Claude client. It never raises: failures come back
    # as a structured, unsuccessful response.
    response = ask_claude(prompt.user, system=prompt.system)

    if not response.success:
        # Log and degrade gracefully; the caller keeps the backend comparison.
        logger.warning(
            "Comparison explanation unavailable (error_type=%s).",
            response.error_type,
        )
        return None

    return response.text
