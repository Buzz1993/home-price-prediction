# ===============================
# src/llm/investment_explanation.py
# ===============================
#
# Phase 15.6 — AI Investment Advisor summary.
#
# Thin orchestration layer that turns the COMBINED structured results of the
# EXISTING backend analysis agents (the Investment Advisor result plus, when
# available, price prediction, valuation, rental and negotiation for the same
# property) into ONE natural-language investment summary.
#
# It performs NO business logic:
#   - It does not predict prices, compute yields, valuations, ROI or risk.
#   - It does not compute an investment score or override the backend's
#     Investment Advisor recommendation.
#   - It never invents numbers, risks, opportunities or properties.
#   - It only reuses the Phase 15.2 Investment Prompt Builder + the Phase 15.1
#     Claude Client to explain the structured results the backend already
#     produced.
#
# Claude is OPTIONAL. If the summary cannot be generated for any reason, this
# module returns None and the caller still returns the backend Investment
# Advisor result unchanged.

import logging

from src.llm.claude_client import ask_claude
from src.llm.prompts import build_investment_prompt

logger = logging.getLogger(__name__)


def explain_investment(analyses: dict) -> str | None:
    """
    Generate a conversational investment summary from combined backend results.

    Args:
        analyses : A mapping of analysis type -> backend result (list of dicts),
                   e.g. { "advisor": [...], "prediction": [...], "rental": [...],
                   "valuation": [...], "negotiation": [...] }. The backend has
                   already produced each of these; this only summarizes them.

    Returns:
        The investment summary text, or None when Claude is unavailable / fails
        or there is nothing to summarize. Returning None lets the caller keep
        the backend Investment Advisor result working without a summary.
    """

    # Nothing to summarize — no analyses at all, or every analysis is empty.
    # Do not spend a Claude call on an empty context.
    if not analyses or not any(analyses.values()):
        return None

    # Reuse the Phase 15.2 builder; never build prompts inline here.
    prompt = build_investment_prompt(analyses)

    # Reuse the Phase 15.1 Claude client. It never raises: failures come back
    # as a structured, unsuccessful response.
    response = ask_claude(prompt.user, system=prompt.system)

    if not response.success:
        # Log and degrade gracefully; the caller keeps the backend result.
        logger.warning(
            "Investment summary unavailable (error_type=%s).",
            response.error_type,
        )
        return None

    return response.text
