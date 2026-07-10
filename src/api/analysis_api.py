# ===============================
# src/api/analysis_api.py
# ===============================

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.mcp.tools.property_tools import (
    compare_properties,
    get_price_prediction,
    get_rental_analysis,
    get_valuation_analysis,
    get_investment_advice,
    get_negotiation_strategy,
    clear_property_analysis_cache,
)
from src.llm.analysis_explanation import explain_analysis
from src.llm.comparison_explanation import explain_comparison
from src.llm.investment_explanation import explain_investment

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analysis",
    tags=["EstateMind Copilot"]
)


# =====================================================
# REQUEST MODEL
# =====================================================

class PropertyIdsRequest(BaseModel):
    property_ids: list[str]


# =====================================================
# COMMON EXECUTOR
# =====================================================

def execute(func, *args, **kwargs):
    """
    Execute EstateMind service functions with
    common exception handling.
    """
    try:
        return func(*args, **kwargs)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =====================================================
# AI ANALYSIS EXPLANATION (Phase 15.4)
# =====================================================

def attach_analysis_explanation(result, analysis_type: str, explain: bool):
    """
    Optionally attach a natural-language explanation to an analysis result.

    The backend analysis output is returned UNCHANGED. When `explain` is
    False (the default) the raw result is returned exactly as before, so the
    existing API contract is preserved. When `explain` is True the result is
    wrapped as `{ "content": <analysis>, "ai_explanation": <text|null> }`,
    where Claude describes what the backend analysis means.

    Claude is optional: if the explanation cannot be generated the analysis is
    still returned (with `ai_explanation` null), so an AI failure never blocks
    the backend analysis.
    """

    if not explain:
        return result

    try:
        explanation = explain_analysis(analysis_type, result)
    except Exception:
        # Never let the explanation layer break a working analysis response.
        logger.exception(
            "Analysis explanation step failed; returning analysis without it."
        )
        explanation = None

    return {"content": result, "ai_explanation": explanation}


# =====================================================
# AI COMPARISON EXPLANATION (Phase 15.5)
# =====================================================

def attach_comparison_explanation(result, explain: bool):
    """
    Optionally attach a natural-language explanation to a comparison result.

    The backend comparison output is returned UNCHANGED. When `explain` is
    False (the default) the raw result is returned exactly as before, so the
    existing API contract is preserved. When `explain` is True the result is
    wrapped as `{ "content": <comparison>, "ai_explanation": <text|null> }`,
    where Claude explains why the backend ranked the properties as it did.

    Claude is optional: if the explanation cannot be generated the comparison
    is still returned (with `ai_explanation` null), so an AI failure never
    blocks the backend comparison. Claude only explains the result — it never
    compares, scores or ranks the properties itself.
    """

    if not explain:
        return result

    try:
        explanation = explain_comparison(result)
    except Exception:
        # Never let the explanation layer break a working comparison response.
        logger.exception(
            "Comparison explanation step failed; returning comparison without it."
        )
        explanation = None

    return {"content": result, "ai_explanation": explanation}


# =====================================================
# AI INVESTMENT SUMMARY (Phase 15.6)
# =====================================================

def _safe_analysis(func, property_ids: list[str]) -> list[dict]:
    """
    Run one existing backend analysis for the investment summary context.

    Used only to GATHER the structured results the backend already produces
    (price prediction, rental, valuation, negotiation) so Claude can summarize
    them together. It performs no analysis itself and degrades to an empty list
    on failure, so a single missing analysis never breaks the summary or the
    primary Investment Advisor result.
    """

    try:
        return func(property_ids) or []
    except Exception:
        logger.exception(
            "Investment summary context step failed (%s); continuing without it.",
            getattr(func, "__name__", "analysis"),
        )
        return []


def attach_investment_summary(advisor_result, property_ids: list[str]):
    """
    Attach a combined natural-language investment summary to the backend
    Investment Advisor result (Phase 15.6).

    The backend Investment Advisor output is returned UNCHANGED under
    `content`. The other EXISTING analyses (price prediction, valuation,
    rental, negotiation) are gathered for the SAME properties and handed to
    Claude, which connects them into one investment summary under
    `ai_explanation`. Claude only summarizes — it never predicts, values,
    scores, ranks or overrides the backend recommendation.

    Claude is optional: if the summary cannot be generated the advisor result
    is still returned (with `ai_explanation` null), so an AI failure never
    blocks the backend Investment Advisor.
    """

    try:
        analyses = {
            "advisor": advisor_result,
            "prediction": _safe_analysis(get_price_prediction, property_ids),
            "valuation": _safe_analysis(get_valuation_analysis, property_ids),
            "rental": _safe_analysis(get_rental_analysis, property_ids),
            "negotiation": _safe_analysis(get_negotiation_strategy, property_ids),
        }
        summary = explain_investment(analyses)
    except Exception:
        # Never let the summary layer break a working advisor response.
        logger.exception(
            "Investment summary step failed; returning advisor result without it."
        )
        summary = None

    return {"content": advisor_result, "ai_explanation": summary}

@router.post("/comparison")
def comparison(request: PropertyIdsRequest, explain: bool = False):
    """
    Compare multiple selected properties.

    The backend comparison agent scores and ranks the properties. When
    `explain=true`, Claude additionally explains that result (Phase 15.5); the
    comparison data itself is never modified.
    """

    if len(request.property_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least two properties are required for comparison."
        )

    result = execute(
        compare_properties,
        request.property_ids
    )

    return attach_comparison_explanation(result, explain)


# =====================================================
# PRICE PREDICTION
# =====================================================

@router.post("/predict")
def predict(
    request: PropertyIdsRequest,
    explain: bool = False,
    analysis_type: str = "prediction",
):
    """
    Predict property price.
    """

    result = execute(
        get_price_prediction,
        request.property_ids
    )

    return attach_analysis_explanation(result, analysis_type, explain)


# =====================================================
# RENTAL ANALYSIS
# =====================================================

@router.post("/rental")
def rental(
    request: PropertyIdsRequest,
    explain: bool = False,
    analysis_type: str = "rental",
):
    """
    Rental yield analysis.
    """

    result = execute(
        get_rental_analysis,
        request.property_ids
    )

    return attach_analysis_explanation(result, analysis_type, explain)


# =====================================================
# VALUATION ANALYSIS
# =====================================================

@router.post("/valuation")
def valuation(
    request: PropertyIdsRequest,
    explain: bool = False,
    analysis_type: str = "valuation",
):
    """
    Fair market valuation.
    """

    result = execute(
        get_valuation_analysis,
        request.property_ids
    )

    return attach_analysis_explanation(result, analysis_type, explain)


# =====================================================
# INVESTMENT ADVISOR
# =====================================================

@router.post("/advisor")
def advisor(
    request: PropertyIdsRequest,
    explain: bool = False,
    analysis_type: str = "advisor",
    summary: bool = False,
):
    """
    Investment advisor analysis.

    The frontend surfaces Risk Analysis from these same rows and can request a
    risk-focused explanation by passing `analysis_type=risk`.

    When `summary=true` (Phase 15.6), the backend additionally combines the
    other existing analyses (price prediction, valuation, rental, negotiation)
    and Claude writes one investment summary over them; the advisor data itself
    is never modified. `summary` takes precedence over `explain`.
    """

    result = execute(
        get_investment_advice,
        request.property_ids
    )

    if summary:
        return attach_investment_summary(result, request.property_ids)

    return attach_analysis_explanation(result, analysis_type, explain)


# =====================================================
# NEGOTIATION STRATEGY
# =====================================================

@router.post("/negotiation")
def negotiation(
    request: PropertyIdsRequest,
    explain: bool = False,
    analysis_type: str = "negotiation",
):
    """
    Negotiation strategy.
    """

    result = execute(
        get_negotiation_strategy,
        request.property_ids
    )

    return attach_analysis_explanation(result, analysis_type, explain)


# =====================================================
# CLEAR CACHE
# =====================================================

@router.post("/cache/clear")
def clear_cache():
    """
    Clear EstateMind enrichment cache.
    """

    return execute(
        clear_property_analysis_cache
    )