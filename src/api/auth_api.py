# ===============================
# src/api/auth_api.py
# ===============================

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(
    tags=["Authentication"]
)


# =====================================================
# REQUEST MODELS
# =====================================================

class SignupRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


# =====================================================
# DEMO USER
# =====================================================

DEMO_USER = {
    "id": "user-001",
    "name": "Demo User",
    "email": "demo@estatemind.ai",
    "password": "password123"
}


# =====================================================
# SIGNUP
# =====================================================

@router.post("/signup")
def signup(request: SignupRequest):
    """
    Demo signup endpoint.

    Replace with database persistence later.
    """

    return {
        "success": True,
        "message": "Account created successfully.",
        "user": {
            "name": request.name,
            "email": request.email
        }
    }


# =====================================================
# LOGIN
# =====================================================

@router.post("/login")
def login(request: LoginRequest):
    """
    Demo login endpoint.
    """

    if (
        request.email != DEMO_USER["email"] or
        request.password != DEMO_USER["password"]
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    return {
        "success": True,
        "token": "demo-session-token",
        "user": {
            "id": DEMO_USER["id"],
            "name": DEMO_USER["name"],
            "email": DEMO_USER["email"]
        }
    }


# =====================================================
# PROFILE
# =====================================================

@router.get("/profile")
def profile():
    """
    Return current user profile.
    """

    return {
        "id": DEMO_USER["id"],
        "name": DEMO_USER["name"],
        "email": DEMO_USER["email"]
    }


# =====================================================
# LOGOUT
# =====================================================

@router.post("/logout")
def logout():
    """
    Logout endpoint.
    """

    return {
        "success": True,
        "message": "Logged out successfully."
    }
