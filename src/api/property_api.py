# ===============================
# src/api/property_api.py
# ===============================

from fastapi import APIRouter, HTTPException

from src.mcp.tools.property_tools import (
    get_property_details,
)

router = APIRouter(
    tags=["Property"]
)


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
# PROPERTY DETAILS
# =====================================================

@router.get("/property/{property_id}")
def property_details(property_id: str):
    """
    Return complete details
    for a single property.
    """

    result = execute(
        get_property_details,
        property_id
    )

    if result.get("error"):
        raise HTTPException(
            status_code=404,
            detail=result.get(
                "error",
                "Property not found."
            )
        )

    return result