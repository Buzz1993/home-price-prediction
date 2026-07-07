# ===============================
# src/api/main.py
# ===============================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.chat_api import router as chat_router
from src.api.property_api import router as property_router
from src.api.analysis_api import router as analysis_router
from src.api.report_api import router as report_router
from src.api.auth_api import router as auth_router
from src.api.saved_api import router as saved_router
from src.api.profile_api import router as profile_router

# =====================================================
# FASTAPI APPLICATION
# =====================================================

app = FastAPI(
    title="EstateMind Copilot API",
    description="REST API for EstateMind property search, chat, analysis, reports, and user services.",
    version="2.0.0"
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