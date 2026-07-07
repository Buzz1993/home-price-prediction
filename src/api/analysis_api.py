# ===============================
# src/api/analysis_api.py
# ===============================

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
# COMPARISON
# =====================================================

@router.post("/comparison")
def comparison(request: PropertyIdsRequest):
    """
    Compare multiple selected properties.
    """

    if len(request.property_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least two properties are required for comparison."
        )

    return execute(
        compare_properties,
        request.property_ids
    )


# =====================================================
# PRICE PREDICTION
# =====================================================

@router.post("/predict")
def predict(request: PropertyIdsRequest):
    """
    Predict property price.
    """

    return execute(
        get_price_prediction,
        request.property_ids
    )


# =====================================================
# RENTAL ANALYSIS
# =====================================================

@router.post("/rental")
def rental(request: PropertyIdsRequest):
    """
    Rental yield analysis.
    """

    return execute(
        get_rental_analysis,
        request.property_ids
    )


# =====================================================
# VALUATION ANALYSIS
# =====================================================

@router.post("/valuation")
def valuation(request: PropertyIdsRequest):
    """
    Fair market valuation.
    """

    return execute(
        get_valuation_analysis,
        request.property_ids
    )


# =====================================================
# INVESTMENT ADVISOR
# =====================================================

@router.post("/advisor")
def advisor(request: PropertyIdsRequest):
    """
    Investment advisor analysis.
    """

    return execute(
        get_investment_advice,
        request.property_ids
    )


# =====================================================
# NEGOTIATION STRATEGY
# =====================================================

@router.post("/negotiation")
def negotiation(request: PropertyIdsRequest):
    """
    Negotiation strategy.
    """

    return execute(
        get_negotiation_strategy,
        request.property_ids
    )


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