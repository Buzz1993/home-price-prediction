# ===============================
# src/llm/report_enhancement.py
# ===============================
#
# Phase 15.10 — AI Report Enhancement (redesigned as the professional
# investment report presenter).
#
# Thin orchestration layer that re-presents the EXISTING backend results as a
# structured investment report. It GATHERS the structured outputs the backend
# already produces for the selected properties (property details, price
# prediction, rental, valuation, advisor — which carries the risk and
# future-growth fields — negotiation and, for 2+ properties, the comparison)
# and hands them, together with the backend-generated report, to the report
# prompt builder. Mirrors the Phase 15.6 investment-summary pattern.
#
# It performs NO business logic:
#   - It does not generate analysis: every value comes from the existing
#     backend services, which are only CALLED here, never reimplemented.
#   - It never invents analysis, prices, recommendations or conclusions, and
#     never changes any fact or figure the backend already produced.
#   - It only reuses the Phase 15.2 Report Prompt Builder + the Phase 15.1
#     LLM client to present what the backend already produced.
#
# The LLM is OPTIONAL. If the report presentation cannot be generated for any
# reason, this module returns None and the caller still returns the backend
# report unchanged, so an AI failure never blocks report generation.

import logging
import re

from src.llm.claude_client import ask_claude, strip_markdown
from src.llm.prompts import build_report_prompt
from src.llm.prompts.report_prompt import DIVIDER_HEAVY, DIVIDER_LIGHT

logger = logging.getLogger(__name__)


# A Markdown table separator row, e.g. "|-----|:---:|----|". Never wanted in
# the plain-text report — the tables keep only their header and data rows.
_TABLE_SEPARATOR_ROW = re.compile(r"^\s*\|?[\s|:\-]+\|?\s*$")

# Any line drawn mostly with divider characters (═ or ━), even when the model
# emitted the wrong length or corrupted it with stray box-drawing characters.
_DIVIDER_LINE = re.compile(r"^\s*[═━]{3}[^\sa-zA-Z0-9]*\s*$")


def _polish_report(text: str) -> str:
    """
    PRESENTATION ONLY: normalize the plain-text layout of the generated
    report. No wording, value or ordering is changed — this only repairs
    visual artifacts so the report renders like a finished document:
      - Markdown table separator rows ('|---|---|') are dropped.
      - Divider lines are redrawn at the exact standard width, keeping the
        heavy (═) or light (━) style the model chose.
      - Runs of 3+ blank lines collapse to one blank line.
    """
    lines = []
    for line in text.splitlines():
        if _TABLE_SEPARATOR_ROW.match(line) and "-" in line:
            continue
        if _DIVIDER_LINE.match(line):
            line = DIVIDER_HEAVY if "═" in line else DIVIDER_LIGHT
        lines.append(line.rstrip())
    polished = "\n".join(lines)
    polished = re.sub(r"\n{3,}", "\n\n", polished)
    return polished.strip()


def _safe(func, *args) -> list | dict:
    """
    Run one existing backend analysis for the report context. Degrades to an
    empty result on failure so one missing analysis never blocks the report.
    """
    try:
        return func(*args) or []
    except Exception:
        logger.exception(
            "Report context step failed (%s); continuing without it.",
            getattr(func, "__name__", "analysis"),
        )
        return []


def _gather_analyses(property_ids: list[str]) -> dict:
    """
    Collect the structured backend analyses for the selected properties by
    calling the EXISTING tools (no analysis is performed here). The advisor
    result also carries the Risk and Future Growth fields.
    """
    # Imported here to keep module import light and avoid cycles.
    from src.mcp.tools import property_tools as tools

    analyses = {
        "overview": _safe(tools.get_properties_by_ids, property_ids),
        "prediction": _safe(tools.get_price_prediction, property_ids),
        "rental": _safe(tools.get_rental_analysis, property_ids),
        "valuation": _safe(tools.get_valuation_analysis, property_ids),
        "advisor": _safe(tools.get_investment_advice, property_ids),
        "negotiation": _safe(tools.get_negotiation_strategy, property_ids),
    }

    # The backend comparison needs at least two properties.
    if len(property_ids) >= 2:
        analyses["comparison"] = _safe(tools.compare_properties, property_ids)

    return analyses


def enhance_report(report: str, property_ids: list[str] | None = None) -> str | None:
    """
    Generate the professional investment-report presentation of the backend
    results for the selected properties.

    Args:
        report       : The report already produced by the backend
                       (create_property_report). Supporting context; the
                       backend owns its content.
        property_ids : The selected property ids. When provided, the existing
                       backend analyses for these properties are gathered and
                       included so the report can summarize them. When omitted
                       (legacy call), the report text alone is re-presented.

    Returns:
        The report text, or None when the LLM is unavailable / fails or there
        is nothing to present. Returning None lets the caller keep the backend
        report working without the enhancement.
    """

    # Nothing to enhance — no report or an empty one. Do not spend an LLM call.
    if not report or not report.strip():
        return None

    analyses = _gather_analyses(property_ids) if property_ids else None

    # Reuse the Phase 15.2 builder; never build prompts inline here.
    prompt = build_report_prompt(report, analyses=analyses)

    # Reuse the Phase 15.1 LLM client. It never raises: failures come back
    # as a structured, unsuccessful response.
    response = ask_claude(prompt.user, system=prompt.system)

    if not response.success:
        # Log and degrade gracefully; the caller keeps the backend report.
        logger.warning(
            "Report enhancement unavailable (error_type=%s).",
            response.error_type,
        )
        return None

    # Presentation only: the report is displayed/exported as plain text, so
    # strip any Markdown syntax the model emitted despite the prompt rules,
    # then repair layout artifacts (table separator rows, malformed divider
    # lines, excess blank lines). Icons, bullets, stars and values untouched.
    return _polish_report(strip_markdown(response.text))
