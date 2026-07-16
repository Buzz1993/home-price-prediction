# ===============================
# src/llm/comparison_report.py
# ===============================
#
# Phase 17.3 — Comparison Report generation for the Property Comparison page.
#
# Thin orchestration layer that re-presents the EXISTING backend comparison
# results as a comparison report matching the /compare page layout. It
# GATHERS the structured outputs the backend already produces for the
# compared properties (comparison winner/rankings, property details, price
# prediction, rental, valuation, advisor — which carries the risk and
# future-growth fields — and negotiation) and hands them to the comparison
# report prompt builder. Mirrors the Phase 15.10 report_enhancement pattern
# and REUSES its helpers — the standard investment report pipeline is
# untouched.
#
# It performs NO business logic:
#   - It does not compare, score or rank: every value comes from the existing
#     backend services, which are only CALLED here, never reimplemented.
#   - It never invents analysis, prices, winners or conclusions, and never
#     changes any fact or figure the backend already produced.
#   - It only reuses the Phase 17.3 Comparison Report Prompt Builder + the
#     Phase 15.1 LLM client to present what the backend already produced.
#
# The LLM is OPTIONAL only in the sense that failures surface as None; the
# API layer decides how to degrade (it falls back to the plain backend
# report so sharing still works).

import logging

from src.llm.claude_client import ask_claude, strip_markdown
from src.llm.prompts import build_comparison_report_prompt

# Reuse the Phase 15.10 helpers verbatim: the same backend-analysis gathering
# and the same plain-text layout polish, so both report types stay identical
# in data sourcing and rendering discipline.
from src.llm.report_enhancement import _gather_analyses, _polish_report

logger = logging.getLogger(__name__)


def generate_comparison_report(
    property_ids: list[str],
    report: str | None = None,
) -> str | None:
    """
    Generate the comparison-report presentation of the backend results for
    the compared properties.

    Args:
        property_ids : The compared property ids (2+). The existing backend
                       analyses for these properties are gathered — including
                       the comparison winner and rankings — and re-presented.
        report       : Optional backend-generated narrative report
                       (create_property_report), passed through as supporting
                       context only.

    Returns:
        The comparison report text, or None when the LLM is unavailable /
        fails. Returning None lets the caller fall back to the backend
        report, so an AI failure never blocks sharing.
    """

    analyses = _gather_analyses(property_ids)

    # Without the backend comparison there is no comparison to report on —
    # the caller falls back to the standard report path.
    comparison = analyses.get("comparison")
    if not isinstance(comparison, dict) or not comparison.get("winner"):
        logger.warning(
            "Comparison report unavailable: the backend comparison returned "
            "no winner for %s.",
            property_ids,
        )
        return None

    # Reuse the prompt builder; never build prompts inline here.
    prompt = build_comparison_report_prompt(analyses, report=report)

    # Reuse the Phase 15.1 LLM client. It never raises: failures come back
    # as a structured, unsuccessful response.
    response = ask_claude(prompt.user, system=prompt.system)

    if not response.success:
        logger.warning(
            "Comparison report unavailable (error_type=%s).",
            response.error_type,
        )
        return None

    # Presentation only: same plain-text polish as the standard report so the
    # existing preview / print / PDF pipeline renders it unchanged.
    return _polish_report(strip_markdown(response.text))
