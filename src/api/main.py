# ===============================
# src/api/main.py
# ===============================

# =====================================================
# WINDOWS UTF-8 CONSOLE FIX
# =====================================================
# On Windows the console defaults to the legacy cp1252 codec, so any
# existing backend logging/print that contains Unicode (emoji) raises
# UnicodeEncodeError. That crashes endpoints executing backend business
# logic before they can return a response.
#
# Reconfigure stdout/stderr to UTF-8 at startup. This preserves all
# existing logging and emoji without touching business logic.
import sys

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="backslashreplace")

import math

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.chat_api import router as chat_router
from src.api.property_api import router as property_router
from src.api.analysis_api import router as analysis_router
from src.api.report_api import router as report_router
from src.api.auth_api import router as auth_router
from src.api.saved_api import router as saved_router
from src.api.profile_api import router as profile_router

# =====================================================
# SAFE JSON RESPONSE
# =====================================================
# Backend services return pandas DataFrame records, which use float NaN for
# missing numeric/text cells (e.g. a property with no amenities). The default
# JSON serializer rejects non-finite floats ("Out of range float values are
# not JSON compliant"), turning any such response into a 500.
#
# Convert non-finite floats (NaN/Infinity) to null at the serialization
# boundary. This is a pure API-layer concern and does not touch business logic.

def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


class SafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return super().render(_json_safe(content))


# =====================================================
# FASTAPI APPLICATION
# =====================================================

app = FastAPI(
    title="EstateMind Copilot API",
    description="REST API for EstateMind property search, chat, analysis, reports, and user services.",
    version="2.0.0",
    default_response_class=SafeJSONResponse,
)

# =====================================================
# CORS
# =====================================================
# Allows the Next.js frontend to call this API.
# Update the allowed origins for production deployment.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "https://estatemind-frontend-production.up.railway.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# ROUTERS
# =====================================================

app.include_router(chat_router)
app.include_router(property_router)
app.include_router(analysis_router)
app.include_router(report_router)
app.include_router(auth_router)
app.include_router(saved_router)
app.include_router(profile_router)

# =====================================================
# ROOT
# =====================================================

@app.get("/", tags=["System"])
def root():
    """
    Root endpoint.
    """
    return {
        "message": "EstateMind Copilot API is running.",
        "version": "2.0.0",
        "status": "healthy",
        "services": [
            "chat",
            "property",
            "analysis",
            "reports",
            "authentication",
            "saved-properties",
            "profile"
        ]
    }


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health", tags=["System"])
def health():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy"
    }