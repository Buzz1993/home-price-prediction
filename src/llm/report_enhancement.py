# ===============================
# src/llm/report_enhancement.py
# ===============================
#
# Phase 15.10 — AI Report Enhancement.
#
# Thin orchestration layer that improves the READABILITY of an EXISTING
# backend-generated report (the Markdown text produced by
# create_property_report). Claude restructures, summarizes and polishes the
# presentation of that report.
#
# It performs NO business logic:
#   - It does not generate the report from scratch or replace the backend
#     report pipeline.
#   - It never invents analysis, prices, recommendations or conclusions, and
#     never changes any fact or figure the backend already produced.
#   - It only reuses the Phase 15.2 Report Prompt Builder + the Phase 15.1
#     Claude Client to re-present the report the backend already produced.
#
# Claude is OPTIONAL. If the enhancement cannot be generated for any reason,
# this module returns None and the caller still returns the backend report
# unchanged, so an AI failure never blocks report generation.

import logging

from src.llm.claude_client import ask_claude
from src.llm.prompts import build_report_prompt

logger = logging.getLogger(__name__)


def enhance_report(report: str) -> str | None:
    """
    Generate a polished, more readable version of a backend-generated report.

    Args:
        report : The Markdown report already produced by the backend
                 (create_property_report). The backend owns the report's
                 content; this only improves how it reads.

    Returns:
        The enhanced report text, or None when Claude is unavailable / fails or
        there is nothing to enhance. Returning None lets the caller keep the
        backend report working without the enhancement.
    """

    # Nothing to enhance — no report or an empty one. Do not spend a Claude call.
    if not report or not report.strip():
        return None

    # Reuse the Phase 15.2 builder; never build prompts inline here.
    prompt = build_report_prompt(report)

    # Reuse the Phase 15.1 Claude client. It never raises: failures come back
    # as a structured, unsuccessful response.
    response = ask_claude(prompt.user, system=prompt.system)

    if not response.success:
        # Log and degrade gracefully; the caller keeps the backend report.
        logger.warning(
            "Report enhancement unavailable (error_type=%s).",
            response.error_type,
        )
        return None

    return response.text
