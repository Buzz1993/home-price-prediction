# ===============================
# src/services/whatsapp_service.py
# ===============================
#
# WhatsApp report delivery through the official Meta WhatsApp Cloud API
# (Phase 16.1). Replaces the former n8n webhook delivery for /report/share.
#
# Responsibilities (delivery only — no report logic lives here):
#
#   1. validate_phone_number()  — normalize + sanity-check the recipient
#   2. generate_report_pdf()    — render the EXISTING report text to a PDF
#                                 (cached, so the same report is only
#                                 rendered once; the text is never
#                                 regenerated here)
#   3. upload_pdf()             — POST /{PHONE_NUMBER_ID}/media  -> media_id
#   4. send_document()          — POST /{PHONE_NUMBER_ID}/messages
#   5. send_report()            — the end-to-end orchestration
#
# Credentials are read from environment variables (.env via python-dotenv):
#
#   WHATSAPP_ACCESS_TOKEN=
#   WHATSAPP_PHONE_NUMBER_ID=
#   WHATSAPP_API_VERSION=v25.0
#
# The service is framework-free: failures raise WhatsAppError with an HTTP
# status code and a meaningful message, and the API layer translates that
# into an HTTPException.

import hashlib
import io
import logging
import os
import re
from xml.sax.saxutils import escape

import requests
from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_FILENAME = "EstateMind Investment Report.pdf"
REQUEST_TIMEOUT = 30  # seconds, per Cloud API request

# EstateMind brand green, matching the frontend primary palette.
BRAND_GREEN = colors.HexColor("#16a34a")


class WhatsAppError(Exception):
    """Delivery failure with an HTTP status code for the API layer."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


# =====================================================
# CONFIGURATION
# =====================================================

def _get_config() -> tuple[str, str, str]:
    """
    Read the Cloud API credentials from the environment at call time (so a
    rotated token is picked up without a restart). Never hardcoded.
    """
    token = (os.getenv("WHATSAPP_ACCESS_TOKEN") or "").strip()
    phone_number_id = (os.getenv("WHATSAPP_PHONE_NUMBER_ID") or "").strip()
    api_version = (os.getenv("WHATSAPP_API_VERSION") or "v25.0").strip()

    if not token or not phone_number_id:
        raise WhatsAppError(
            "WhatsApp delivery is not configured. Set WHATSAPP_ACCESS_TOKEN "
            "and WHATSAPP_PHONE_NUMBER_ID in the backend environment.",
            status_code=503,
        )

    return token, phone_number_id, api_version


def _api_url(endpoint: str) -> str:
    _, phone_number_id, api_version = _get_config()
    return f"https://graph.facebook.com/{api_version}/{phone_number_id}/{endpoint}"


def _auth_header() -> dict:
    token, _, _ = _get_config()
    return {"Authorization": f"Bearer {token}"}


# =====================================================
# PHONE NUMBER VALIDATION
# =====================================================

def validate_phone_number(phone_number: str) -> str:
    """
    Normalize a phone number to the digits-only international format the
    Cloud API expects (e.g. "+91 98765 43210" -> "919876543210") and reject
    values that cannot be a valid number. Formatting only — no country-code
    guessing is performed.
    """
    raw = (phone_number or "").strip()
    if not raw:
        raise WhatsAppError("Phone number is required.", status_code=400)

    normalized = re.sub(r"[\s\-().]", "", raw)
    if normalized.startswith("+"):
        normalized = normalized[1:]

    if not normalized.isdigit() or not 8 <= len(normalized) <= 15:
        raise WhatsAppError(
            "Invalid phone number. Use the international format, "
            "e.g. +91 98765 43210.",
            status_code=400,
        )

    return normalized


# =====================================================
# PDF GENERATION (existing report text -> PDF, cached)
# =====================================================

# The same generated report is often shared to several numbers; cache the
# rendered PDF by content hash so it is only built once. Bounded so a
# long-running server never accumulates stale reports.
_pdf_cache: dict[str, bytes] = {}
_PDF_CACHE_LIMIT = 8

# Inline Markdown emphasis -> ReportLab inline tags (applied after escaping).
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*([^*]+?)\*(?!\*)")


def _inline_markup(text: str) -> str:
    escaped = escape(text)
    escaped = _BOLD.sub(r"<b>\1</b>", escaped)
    escaped = _ITALIC.sub(r"<i>\1</i>", escaped)
    return escaped


def _pdf_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "EMTitle", parent=base["Title"], textColor=BRAND_GREEN,
            fontSize=20, leading=24, spaceAfter=4,
        ),
        "h1": ParagraphStyle(
            "EMH1", parent=base["Heading1"], textColor=BRAND_GREEN,
            fontSize=15, leading=19, spaceBefore=12, spaceAfter=4,
        ),
        "h2": ParagraphStyle(
            "EMH2", parent=base["Heading2"], textColor=colors.HexColor("#166534"),
            fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "EMBody", parent=base["BodyText"], fontSize=10, leading=14.5,
            spaceAfter=4,
        ),
        "meta": ParagraphStyle(
            "EMMeta", parent=base["BodyText"], fontSize=8.5, leading=11,
            textColor=colors.HexColor("#6b7280"),
        ),
    }


def generate_report_pdf(report_text: str) -> bytes:
    """
    Render the EXISTING Markdown report text to a clean, branded PDF. The
    report content is used verbatim (headings, bullets and emphasis are only
    reformatted for print) — nothing is regenerated or rewritten. Results are
    cached by content hash so each report is rendered at most once.
    """
    text = (report_text or "").strip()
    if not text:
        raise WhatsAppError("There is no report to send.", status_code=400)

    cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cached = _pdf_cache.get(cache_key)
    if cached is not None:
        return cached

    styles = _pdf_styles()
    story = [
        Paragraph("EstateMind Investment Report", styles["title"]),
        HRFlowable(width="100%", thickness=2, color=BRAND_GREEN, spaceAfter=10),
    ]

    bullets: list[ListItem] = []

    def flush_bullets():
        if bullets:
            story.append(
                ListFlowable(
                    list(bullets), bulletType="bullet",
                    leftIndent=14, bulletFontSize=8,
                )
            )
            bullets.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line or re.fullmatch(r"[-*_]{3,}", line):
            flush_bullets()
            if line:
                story.append(Spacer(1, 4))
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            flush_bullets()
            level, content = heading.groups()
            style = styles["h1"] if len(level) == 1 else styles["h2"]
            story.append(Paragraph(_inline_markup(content), style))
            continue

        bullet = re.match(r"^[-*+]\s+(.*)$", line)
        if bullet:
            bullets.append(
                ListItem(Paragraph(_inline_markup(bullet.group(1)), styles["body"]))
            )
            continue

        flush_bullets()
        story.append(Paragraph(_inline_markup(line), styles["body"]))

    flush_bullets()
    story.append(Spacer(1, 10))
    story.append(
        Paragraph("Generated by EstateMind — AI Real Estate Copilot", styles["meta"])
    )

    buffer = io.BytesIO()
    SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="EstateMind Investment Report",
        author="EstateMind",
    ).build(story)

    pdf_bytes = buffer.getvalue()

    if len(_pdf_cache) >= _PDF_CACHE_LIMIT:
        _pdf_cache.pop(next(iter(_pdf_cache)))
    _pdf_cache[cache_key] = pdf_bytes

    return pdf_bytes


# =====================================================
# CLOUD API CALLS
# =====================================================

def _meta_error_message(response: requests.Response) -> str:
    """Pull the human-readable error out of a Graph API error response."""
    try:
        error = response.json().get("error") or {}
    except ValueError:
        error = {}

    message = error.get("message") or response.text[:200] or "Unknown error"

    # Expired/invalid token surfaces as an OAuthException (usually HTTP 401).
    if response.status_code == 401 or error.get("type") == "OAuthException":
        return (
            "WhatsApp access token is invalid or expired. "
            f"Update WHATSAPP_ACCESS_TOKEN. ({message})"
        )
    return message


def _post(url: str, *, step: str, **kwargs) -> dict:
    """POST to the Cloud API with shared timeout / network error handling."""
    try:
        response = requests.post(url, timeout=REQUEST_TIMEOUT, **kwargs)
    except requests.Timeout:
        raise WhatsAppError(
            f"WhatsApp {step} timed out. Please try again.", status_code=504
        )
    except requests.RequestException as e:
        raise WhatsAppError(
            f"Could not reach the WhatsApp API ({step}): {e}", status_code=502
        )

    if response.status_code >= 400:
        message = _meta_error_message(response)
        logger.error("WhatsApp %s failed (%s): %s", step, response.status_code, message)
        raise WhatsAppError(f"WhatsApp {step} failed: {message}", status_code=502)

    try:
        return response.json()
    except ValueError:
        raise WhatsAppError(
            f"WhatsApp {step} returned an unexpected response.", status_code=502
        )


def upload_pdf(pdf_bytes: bytes, filename: str = DEFAULT_FILENAME) -> str:
    """
    Upload a PDF to the Cloud API media endpoint and return its media_id.

    POST https://graph.facebook.com/{v}/{PHONE_NUMBER_ID}/media
    multipart/form-data: messaging_product=whatsapp, file=<pdf>
    """
    if not pdf_bytes:
        raise WhatsAppError("Report PDF is missing or empty.", status_code=500)

    payload = _post(
        _api_url("media"),
        step="media upload",
        headers=_auth_header(),
        data={"messaging_product": "whatsapp"},
        files={"file": (filename, pdf_bytes, "application/pdf")},
    )

    media_id = payload.get("id")
    if not media_id:
        raise WhatsAppError(
            "WhatsApp media upload did not return a media id.", status_code=502
        )
    return media_id


def send_document(
    phone_number: str,
    media_id: str,
    filename: str = DEFAULT_FILENAME,
) -> dict:
    """
    Send an uploaded document to a WhatsApp number.

    POST https://graph.facebook.com/{v}/{PHONE_NUMBER_ID}/messages
    """
    payload = _post(
        _api_url("messages"),
        step="message send",
        headers={**_auth_header(), "Content-Type": "application/json"},
        json={
            "messaging_product": "whatsapp",
            "to": phone_number,
            "type": "document",
            "document": {"id": media_id, "filename": filename},
        },
    )

    messages = payload.get("messages") or []
    message_id = messages[0].get("id") if messages else None
    return {"message_id": message_id, "response": payload}


# =====================================================
# END-TO-END DELIVERY
# =====================================================

def send_report(
    phone_number: str,
    report_text: str,
    filename: str = DEFAULT_FILENAME,
) -> dict:
    """
    Deliver an EXISTING report to a WhatsApp number:
    validate the number -> render (or reuse the cached) PDF -> upload ->
    send as a document. Returns the ShareResult shape the frontend already
    consumes ({status, status_code}), plus the WhatsApp message id.
    """
    recipient = validate_phone_number(phone_number)
    pdf_bytes = generate_report_pdf(report_text)
    media_id = upload_pdf(pdf_bytes, filename)
    sent = send_document(recipient, media_id, filename)

    logger.info(
        "WhatsApp report delivered to %s (message id %s)",
        recipient,
        sent.get("message_id"),
    )

    return {
        "status": "success",
        "status_code": 200,
        "message_id": sent.get("message_id"),
    }
