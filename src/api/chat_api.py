# ===============================
# src/api/chat_api.py
# ===============================

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.services.chat_service import parse_intent_and_execute
from src.llm.search_explanation import explain_search_results

logger = logging.getLogger(__name__)

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
# AI SEARCH EXPLANATION (Phase 15.3)
# =====================================================

def attach_search_explanation(result, user_query: str):
    """
    Attach an optional natural-language explanation to SEARCH results.

    The backend search, ranking and recommendation output is returned
    UNCHANGED — this only adds an optional `ai_explanation` field describing
    why the backend recommended those properties. Claude is optional: if the
    explanation cannot be generated the results are returned as-is.
    """

    # Only search results are explained; every other response is untouched.
    if not isinstance(result, dict) or result.get("type") != "search_results":
        return result

    query_state = result.get("current_query_state") or {}

    try:
        explanation = explain_search_results(
            user_query=user_query,
            results=result.get("content") or [],
            filters=query_state.get("active_filters"),
            weights=query_state.get("chat_preference_weights"),
        )
    except Exception:
        # Never let the explanation layer break a working search response.
        logger.exception(
            "Search explanation step failed; returning results without it."
        )
        explanation = None

    if explanation:
        result["ai_explanation"] = explanation

    return result


# =====================================================
# CHAT
# =====================================================

@router.post("/chat")
def chat(request: ChatRequest):
    """
    EstateMind Copilot endpoint.

    Routes the user request to the existing backend chat workflow without
    duplicating any business logic. For search responses, Claude additionally
    explains why the backend recommended the returned properties (Phase 15.3);
    the search results themselves are never modified.
    """

    result = execute(
        parse_intent_and_execute,
        user_prompt=request.message,
        session_state_tray=request.staged_property_ids,
        current_ui_sliders=request.slider_weights,
    )

    return attach_search_explanation(result, request.message)
