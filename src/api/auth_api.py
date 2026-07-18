# ===============================
# src/api/auth_api.py
# ===============================
#
# Authentication endpoints for the Next.js frontend (Phase 18.18).
#
# Same four endpoints and response shapes as before (POST /signup,
# POST /login, GET /profile, POST /logout) — only the demo-stub internals
# were replaced. Accounts and session tokens live in a small JSON file so
# they survive --reload restarts. Standard library only: passwords are
# salted PBKDF2-SHA256 hashes, tokens are opaque random strings.

import json
import re
import secrets
import threading
import uuid
from hashlib import pbkdf2_hmac
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException
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
# USER STORE
# =====================================================
# cache/ already holds backend runtime state; users.json follows the same
# pattern. Kept outside src/ so writes do not trigger uvicorn --reload.

STORE_PATH = Path(__file__).resolve().parents[2] / "cache" / "users.json"
_store_lock = threading.Lock()

PBKDF2_ITERATIONS = 100_000


def _load_store() -> dict:
    if STORE_PATH.exists():
        try:
            with STORE_PATH.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError):
            pass
    return {"users": [], "sessions": {}}


def _save_store(store: dict) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STORE_PATH.open("w", encoding="utf-8") as file:
        json.dump(store, file, indent=2)


def _hash_password(password: str, salt: str) -> str:
    return pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS
    ).hex()


def _public_user(user: dict) -> dict:
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
    }


def _validate_password(password: str) -> str | None:
    """Return a user-friendly error message, or None when the password is valid."""
    problems = []
    if len(password) < 8:
        problems.append("minimum 8 characters")
    if not re.search(r"[A-Za-z]", password):
        problems.append("one letter")
    if not re.search(r"\d", password):
        problems.append("one number")
    if not re.search(r"[^A-Za-z0-9]", password):
        problems.append("one special character")
    if problems:
        return "Password must contain: " + ", ".join(problems) + "."
    return None


def _user_for_token(authorization: str | None) -> dict:
    """Resolve the Authorization: Bearer header to a stored user or raise 401."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated.")
    token = authorization.removeprefix("Bearer ").strip()

    with _store_lock:
        store = _load_store()
        user_id = store["sessions"].get(token)
        user = next(
            (u for u in store["users"] if u["id"] == user_id), None
        )

    if user is None:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    return user


# =====================================================
# SIGNUP
# =====================================================

@router.post("/signup")
def signup(request: SignupRequest):
    """
    Create a new account and start a session.
    """

    name = request.name.strip()
    email = request.email.strip().lower()

    if not name:
        raise HTTPException(status_code=400, detail="Name is required.")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")

    password_error = _validate_password(request.password)
    if password_error:
        raise HTTPException(status_code=400, detail=password_error)

    with _store_lock:
        store = _load_store()

        if any(u["email"] == email for u in store["users"]):
            raise HTTPException(
                status_code=409,
                detail="This email is already registered."
            )

        salt = secrets.token_hex(16)
        user = {
            "id": f"user-{uuid.uuid4().hex[:8]}",
            "name": name,
            "email": email,
            "salt": salt,
            "password_hash": _hash_password(request.password, salt),
        }
        token = secrets.token_urlsafe(32)

        store["users"].append(user)
        store["sessions"][token] = user["id"]
        _save_store(store)

    return {
        "success": True,
        "message": "Account created successfully.",
        "token": token,
        "user": _public_user(user),
    }


# =====================================================
# LOGIN
# =====================================================

@router.post("/login")
def login(request: LoginRequest):
    """
    Authenticate a user and start a session.
    """

    email = request.email.strip().lower()

    with _store_lock:
        store = _load_store()
        user = next((u for u in store["users"] if u["email"] == email), None)

        if user is None or not secrets.compare_digest(
            user["password_hash"], _hash_password(request.password, user["salt"])
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password."
            )

        token = secrets.token_urlsafe(32)
        store["sessions"][token] = user["id"]
        _save_store(store)

    return {
        "success": True,
        "token": token,
        "user": _public_user(user),
    }


# =====================================================
# PROFILE
# =====================================================

@router.get("/profile")
def profile(authorization: str | None = Header(default=None)):
    """
    Return the authenticated user's profile.
    """

    return _public_user(_user_for_token(authorization))


# =====================================================
# LOGOUT
# =====================================================

@router.post("/logout")
def logout(authorization: str | None = Header(default=None)):
    """
    End the current session.
    """

    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        with _store_lock:
            store = _load_store()
            if token in store["sessions"]:
                del store["sessions"][token]
                _save_store(store)

    return {
        "success": True,
        "message": "Logged out successfully."
    }
