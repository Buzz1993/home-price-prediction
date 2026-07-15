# ===============================
# src/services/pdf_service.py
# ===============================
#
# The ONE PDF generator in the application (Phase 16.2).
#
# Renders the EXISTING Next.js report page (/print/report) with headless
# Chromium (Playwright) and exports it as an A4 PDF — the exact same
# document the browser's Export PDF / Chrome print flow produces, with all
# Tailwind styling, cards, tables, icons, colors and page breaks intact.
# Nothing is redrawn manually: the frontend's own print CSS
# (frontend/app/globals.css, .report-print-root) does the layout, exactly
# as it does in the browser.
#
# Used by:
#   - POST /report/pdf      (Export / Download PDF in the frontend)
#   - POST /report/share    (WhatsApp delivery — whatsapp_service.py)
#
# so every delivery channel ships the identical, pixel-accurate PDF.
#
# Flow:
#   report text -> inject as window.__ESTATEMIND_REPORT__ ->
#   open FRONTEND_BASE_URL/print/report -> wait for render + fonts +
#   network idle -> page.pdf(A4, printBackground, preferCSSPageSize)
#
# Configuration (.env):
#   FRONTEND_BASE_URL=http://localhost:3000   (the running Next.js app)

import hashlib
import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

PRINT_ROUTE = "/print/report"
RENDER_TIMEOUT_MS = 60_000  # per navigation/selector wait

# The same generated report is often exported and shared several times;
# cache the rendered PDF by content hash so Chromium only renders it once.
# Bounded so a long-running server never accumulates stale reports.
_pdf_cache: dict[str, bytes] = {}
_PDF_CACHE_LIMIT = 8


class PdfRenderError(Exception):
    """PDF rendering failure with an HTTP status code for the API layer."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


def _frontend_base_url() -> str:
    return (os.getenv("FRONTEND_BASE_URL") or "http://localhost:3000").rstrip("/")


def generate_report_pdf(report_text: str) -> bytes:
    """
    Render the EXISTING report text to a PDF through the EXISTING Next.js
    print route, using headless Chromium. The report content is used
    verbatim — nothing is regenerated or restyled here. Results are cached
    by content hash so each report is rendered at most once.
    """
    text = (report_text or "").strip()
    if not text:
        raise PdfRenderError("There is no report to render.", status_code=400)

    cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cached = _pdf_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise PdfRenderError(
            "PDF rendering is not available: Playwright is not installed. "
            "Run: .venv2\\Scripts\\python.exe -m pip install playwright "
            "&& .venv2\\Scripts\\python.exe -m playwright install chromium",
            status_code=503,
        )

    url = f"{_frontend_base_url()}{PRINT_ROUTE}"

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()

                # Hand the already-generated report to the print route before
                # any page script runs, so the page renders exactly this text.
                page.add_init_script(
                    f"window.__ESTATEMIND_REPORT__ = {json.dumps(text)};"
                )

                # Wait for network idle so styles, fonts and icons are loaded.
                page.goto(url, wait_until="networkidle", timeout=RENDER_TIMEOUT_MS)

                # The route flags the fully-rendered document (the print copy
                # is display:none on screen, so wait for attached, not visible).
                page.wait_for_selector(
                    "[data-report-ready]", state="attached", timeout=RENDER_TIMEOUT_MS
                )

                # Ensure every webfont is loaded before printing.
                page.evaluate("document.fonts.ready")

                # A4 portrait, backgrounds on, CSS-defined page size/margins
                # (@page in globals.css) — identical to Chrome's print output.
                pdf_bytes = page.pdf(
                    format="A4",
                    landscape=False,
                    print_background=True,
                    prefer_css_page_size=True,
                )
            finally:
                browser.close()
    except PlaywrightError as e:
        message = str(e).splitlines()[0] if str(e) else "Unknown Chromium error"
        if "Executable doesn't exist" in message:
            raise PdfRenderError(
                "PDF rendering is not available: the Chromium browser is not "
                "installed. Run: .venv2\\Scripts\\python.exe -m playwright "
                "install chromium",
                status_code=503,
            )
        if "ERR_CONNECTION_REFUSED" in message:
            raise PdfRenderError(
                f"PDF rendering failed: the frontend is not reachable at {url}. "
                "Start the Next.js app (npm run dev) or set FRONTEND_BASE_URL.",
                status_code=503,
            )
        logger.error("Report PDF rendering failed: %s", message)
        raise PdfRenderError(f"PDF rendering failed: {message}", status_code=502)

    if not pdf_bytes:
        raise PdfRenderError("PDF rendering produced an empty file.", status_code=502)

    if len(_pdf_cache) >= _PDF_CACHE_LIMIT:
        _pdf_cache.pop(next(iter(_pdf_cache)))
    _pdf_cache[cache_key] = pdf_bytes

    return pdf_bytes
