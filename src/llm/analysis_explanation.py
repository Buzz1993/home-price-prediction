# ===============================
# src/llm/analysis_explanation.py
# ===============================
#
# Phase 15.4 — AI Property Analysis explanation.
#
# Thin orchestration layer that turns an EXISTING backend analysis result
# (risk, rental, valuation, negotiation, price prediction, investment advice)
# into a natural-language explanation of what the analysis means.
#
# It performs NO business logic:
#   - It does not predict prices, compute yields, valuations, ROI or risk.
#   - It never invents numbers, recommendations or properties.
#   - It only reuses the Phase 15.2 Analysis Prompt Builder + the Phase 15.1
#     Claude Client to explain the structured results the backend already
#     produced.
#
# Claude is OPTIONAL. If the explanation cannot be generated for any reason,
# this module returns None and the caller still returns the backend analysis
# unchanged.

import logging

from src.llm.claude_client import ask_claude, strip_markdown
from src.llm.prompts import build_analysis_prompt

logger = logging.getLogger(__name__)


def explain_analysis(
    analysis_type: str,
    analysis: list[dict],
) -> str | None:
    """
    Generate a conversational explanation for a backend analysis result.

    Args:
        analysis_type : The kind of analysis to explain (risk, rental,
                        valuation, negotiation, prediction, advisor, ...).
                        Used only to label the explanation; it never changes
                        the underlying data.
        analysis      : The structured analysis records already produced by
                        the backend (list of dicts, one per property).

    Returns:
        The explanation text, or None when Claude is unavailable / fails or
        there is nothing to explain. Returning None lets the caller keep the
        backend analysis working without an explanation.
    """

    # Nothing to explain — do not spend a Claude call on an empty result.
    if not analysis:
        return None

    # Reuse the Phase 15.2 builder; never build prompts inline here.
    prompt = build_analysis_prompt(
        analysis_type=analysis_type,
        analysis=analysis,
    )

    # Reuse the Phase 15.1 Claude client. It never raises: failures come back
    # as a structured, unsuccessful response.
    response = ask_claude(prompt.user, system=prompt.system)

    if not response.success:
        # Log and degrade gracefully; the caller keeps the backend analysis.
        logger.warning(
            "Analysis explanation unavailable (analysis_type=%s, error_type=%s).",
            analysis_type,
            response.error_type,
        )
        return None

    return strip_markdown(response.text)


#===============================================================================================================================================================

