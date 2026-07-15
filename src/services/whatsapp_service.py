# ===============================
# src/services/whatsapp_service.py
# ===============================
#
# WhatsApp report delivery through the official Meta WhatsApp Cloud API
# (Phase 16.1). Replaces the former n8n webhook delivery for /report/share.
#
# Responsibilities (delivery only — no report or PDF logic lives here):
#
#   1. validate_phone_number()  — normalize + sanity-check the recipient
#   2. upload_pdf()             — POST /{PHONE_NUMBER_ID}/media  -> media_id
#   3. send_document()          — POST /{PHONE_NUMBER_ID}/messages
#   4. send_report()            — the end-to-end orchestration
#
# The PDF itself comes from the application's single PDF generator
# (src/services/pdf_service.py, Phase 16.2): headless Chromium renders the
# EXISTING Next.js report page, so the document sent on WhatsApp is
# pixel-identical to the Export PDF / browser-print output. The former
# ReportLab renderer is gone.
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

import logging
import os
import re

import requests
from dotenv import load_dotenv

from src.services.pdf_service import PdfRenderError, generate_report_pdf

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_FILENAME = "EstateMind Investment Report.pdf"
REQUEST_TIMEOUT = 30  # seconds, per Cloud API request


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
    validate the number -> render (or reuse the cached) PDF through the
    application's single Chromium-based generator (pdf_service) -> upload ->
    send as a document. Returns the ShareResult shape the frontend already
    consumes ({status, status_code}), plus the WhatsApp message id.
    """
    recipient = validate_phone_number(phone_number)
    try:
        pdf_bytes = generate_report_pdf(report_text)
    except PdfRenderError as e:
        raise WhatsAppError(str(e), status_code=e.status_code)
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
