# ===============================
# src/api/report_api.py
# ===============================

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.mcp.tools.property_tools import (
    create_property_report,
    send_property_report,
)
from src.llm.report_enhancement import enhance_report

logger = logging.getLogger(__name__)

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
# AI REPORT ENHANCEMENT (Phase 15.10)
# =====================================================

def attach_report_enhancement(report, enhance: bool):
    """
    Optionally attach a Claude-enhanced version of a backend-generated report.

    The backend report is returned UNCHANGED. When `enhance` is False (the
    default) the raw report string is returned exactly as before, so the
    existing API contract is preserved. When `enhance` is True the result is
    wrapped as `{ "content": <backend report>, "ai_enhanced": <text|null> }`,
    where Claude improves the readability and presentation of the SAME report.

    Claude is optional: if the enhancement cannot be generated the backend
    report is still returned (with `ai_enhanced` null), so an AI failure never
    blocks report generation. Claude only re-presents the backend report — it
    never generates, calculates or invents any report content itself.
    """

    if not enhance:
        return report

    try:
        enhanced = enhance_report(report)
    except Exception:
        # Never let the enhancement layer break a working report response.
        logger.exception(
            "Report enhancement step failed; returning backend report without it."
        )
        enhanced = None

    return {"content": report, "ai_enhanced": enhanced}


# =====================================================
# GENERATE REPORT
# =====================================================

@router.post("/report")
def generate_report(request: ReportRequest, enhance: bool = False):
    """
    Generate AI report
    for selected properties.

    The backend composes the report. When `enhance=true` (Phase 15.10), Claude
    additionally produces a more readable, better-structured version of that
    SAME report; the backend report content itself is never modified.
    """

    if not request.property_ids:
        raise HTTPException(
            status_code=400,
            detail="At least one property is required."
        )

    report = execute(
        create_property_report,
        request.property_ids
    )

    return attach_report_enhancement(report, enhance)


# =====================================================
# SHARE REPORT
# =====================================================

@router.post("/report/share")
def share_report(request: ShareReportRequest, enhance: bool = False):
    """
    Generate and share a property report
    through WhatsApp/SMS.

    Reuses the existing sharing workflow (MCP tool send_property_report + n8n)
    unchanged. When `enhance=true` (Phase 15.10), the SAME backend report is
    first passed through Claude for readability so the shared report matches the
    enhanced preview; if the enhancement is unavailable the backend report is
    shared as-is, so sharing never breaks.
    """

    report = execute(
        create_property_report,
        request.property_ids
    )

    if enhance:
        try:
            enhanced = enhance_report(report)
        except Exception:
            logger.exception(
                "Report enhancement step failed; sharing backend report without it."
            )
            enhanced = None
        if enhanced:
            report = enhanced

    return execute(
        send_property_report,
        request.phone_number,
        report
    )