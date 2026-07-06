# ===============================
# src/api/main.py
# ===============================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.copilot_api import router as copilot_router

# =====================================================
# FASTAPI APPLICATION
# =====================================================

app = FastAPI(
    title="EstateMind Copilot API",
    description="REST API for EstateMind Copilot property analysis services.",
    version="1.0.0"
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

app.include_router(copilot_router)

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
        "version": "1.0.0",
        "status": "healthy"
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