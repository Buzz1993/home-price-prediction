# ===============================
# src/api/report_api.py
# ===============================

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from src.mcp.tools.property_tools import create_property_report
from src.llm.report_enhancement import enhance_report
from src.services.pdf_service import PdfRenderError, generate_report_pdf
from src.services.whatsapp_service import WhatsAppError, send_report

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Reports"]
)


# =====================================================
# REQUEST MODELS
# =====================================================

class ReportRequest(BaseModel):
    property_ids: list[str]


class ReportPdfRequest(BaseModel):
    # The already-generated report text from the frontend preview. The PDF is
    # always rendered from exactly this text — nothing is regenerated.
    report: str


class ShareReportRequest(BaseModel):
    property_ids: list[str]
    phone_number: str
    # The already-generated report text from the frontend preview (Phase 16.1).
    # When provided, the SAME report the user is looking at is delivered and
    # nothing is regenerated; when omitted, the report is generated once here
    # (the pre-16.1 behavior), so existing callers keep working.
    report: str | None = None


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

def attach_report_enhancement(report, enhance: bool, property_ids: list[str] | None = None):
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
        enhanced = enhance_report(report, property_ids)
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

    return attach_report_enhancement(report, enhance, request.property_ids)


# =====================================================
# REPORT PDF (Phase 16.2 — single Chromium PDF pipeline)
# =====================================================

@router.post("/report/pdf")
def report_pdf(request: ReportPdfRequest):
    """
    Render the previewed report to a PDF and return it for download.

    This is the SAME generator WhatsApp delivery uses
    (src/services/pdf_service.py): headless Chromium prints the existing
    Next.js report page, so Export PDF, Download PDF and the WhatsApp
    document are always pixel-identical. No report content is generated or
    modified here.
    """

    try:
        pdf_bytes = generate_report_pdf(request.report)
    except PdfRenderError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
                'attachment; filename="EstateMind Investment Report.pdf"'
        },
    )


# =====================================================
# SHARE REPORT
# =====================================================

@router.post("/report/share")
def share_report(request: ShareReportRequest, enhance: bool = False):
    """
    Share a property report to a WhatsApp number through the official Meta
    WhatsApp Cloud API (Phase 16.1): the report is rendered to a PDF once,
    uploaded to the Cloud API media endpoint and sent as a document message.

    The report text is NEVER regenerated when the frontend passes the
    previewed report in `request.report`. Only when it is absent is the
    report generated here (with the Phase 15.10 Claude readability pass when
    `enhance=true`), preserving the pre-16.1 contract for existing callers.
    """

    report = (request.report or "").strip()

    if not report:
        report = execute(
            create_property_report,
            request.property_ids
        )

        if enhance:
            try:
                enhanced = enhance_report(report, request.property_ids)
            except Exception:
                logger.exception(
                    "Report enhancement step failed; sharing backend report without it."
                )
                enhanced = None
            if enhanced:
                report = enhanced

    try:
        return send_report(request.phone_number, report)
    except WhatsAppError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))