# ===============================
# src/api/chat_api.py
# ===============================

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.services.chat_service import parse_intent_and_execute

router = APIRouter(tags=["Chat"])


# =====================================================
# REQUEST MODEL
# =====================================================

class ChatRequest(BaseModel):
    message: str
    staged_property_ids: list[str] = Field(default_factory=list)
    slider_weights: dict | None = Field(default=None)


# =====================================================
# COMMON EXECUTOR
# =====================================================

def execute(func, *args, **kwargs):
    """
    Execute service functions with common exception handling.
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
# CHAT
# =====================================================

@router.post("/chat")
def chat(request: ChatRequest):
    """
    EstateMind Copilot endpoint.

    Routes the user request to the existing backend
    chat workflow without duplicating any business logic.
    """

    return execute(
        parse_intent_and_execute,
        user_prompt=request.message,
        session_state_tray=request.staged_property_ids,
        current_ui_sliders=request.slider_weights,
    )