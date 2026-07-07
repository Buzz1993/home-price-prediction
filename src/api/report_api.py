# ===============================
# src/api/report_api.py
# ===============================

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.mcp.tools.property_tools import (
    create_property_report,
    send_property_report,
)

router = APIRouter(
    tags=["Reports"]
)


# =====================================================
# REQUEST MODELS
# =====================================================

class ReportRequest(BaseModel):
    property_ids: list[str]


class ShareReportRequest(BaseModel):
    property_ids: list[str]
    phone_number: str


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
# GENERATE REPORT
# =====================================================

@router.post("/report")
def generate_report(request: ReportRequest):
    """
    Generate AI report
    for selected properties.
    """

    if not request.property_ids:
        raise HTTPException(
            status_code=400,
            detail="At least one property is required."
        )

    return execute(
        create_property_report,
        request.property_ids
    )


# =====================================================
# SHARE REPORT
# =====================================================

@router.post("/report/share")
def share_report(request: ShareReportRequest):
    """
    Generate and share a property report
    through WhatsApp/SMS.
    """

    report = execute(
        create_property_report,
        request.property_ids
    )

    return execute(
        send_property_report,
        request.phone_number,
        report
    )