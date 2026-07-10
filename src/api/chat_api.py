# ===============================
# src/api/chat_api.py
# ===============================

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.services.chat_service import parse_intent_and_execute
from src.llm.search_explanation import explain_search_results
from src.llm.conversation_memory import get_session_memory_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


# =====================================================
# REQUEST MODEL
# =====================================================

class ChatRequest(BaseModel):
    message: str
    staged_property_ids: list[str] = Field(default_factory=list)
    slider_weights: dict | None = Field(default=None)
    # Optional session identifier for conversational memory (Phase 15.7). When
    # provided, the backend keeps lightweight, session-scoped context so Claude
    # can understand follow-up questions. When omitted, the endpoint behaves
    # exactly as before (stateless), so the API contract is preserved.
    session_id: str | None = Field(default=None)


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

def attach_search_explanation(result, user_query: str, memory: str | None = None):
    """
    Attach an optional natural-language explanation to SEARCH results.

    The backend search, ranking and recommendation output is returned
    UNCHANGED — this only adds an optional `ai_explanation` field describing
    why the backend recommended those properties. Claude is optional: if the
    explanation cannot be generated the results are returned as-is.

    `memory` is an optional session conversation-memory summary (Phase 15.7)
    passed through as context only; it never changes the search results.
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
            memory=memory,
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
    explains why the backend recommended the returned properties (Phase 15.3).

    Conversational memory (Phase 15.7): when the request carries a `session_id`,
    lightweight session-scoped context is kept so the existing backend follow-up
    logic works across HTTP requests and Claude can resolve references in
    follow-up questions. Memory is best-effort context only — it never performs
    business logic, and if it is unavailable the endpoint still answers using
    the current request and backend response.
    """

    # -------------------------------------------------
    # Load session memory (best-effort; never blocks the response).
    # -------------------------------------------------
    memory = None
    session_state = None
    memory_summary = None
    try:
        memory = get_session_memory_store().get(request.session_id)
        # The backend workflow already reads/writes its own follow-up state on
        # this dict (last_search_filters, search_page, ...). Keeping it alive
        # between requests is what enables multi-turn follow-ups.
        session_state = memory.session_state
        memory.record_user(request.message)
        # Build the memory context BEFORE this turn's answer is recorded so the
        # prompt reflects prior context, not the reply we are about to produce.
        memory_summary = memory.summary()
    except Exception:
        logger.exception("Conversation memory unavailable; continuing stateless.")
        memory = None
        session_state = None
        memory_summary = None

    result = execute(
        parse_intent_and_execute,
        user_prompt=request.message,
        session_state_tray=request.staged_property_ids,
        current_ui_sliders=request.slider_weights,
        session_state=session_state,
    )

    result = attach_search_explanation(result, request.message, memory=memory_summary)

    # -------------------------------------------------
    # Record the assistant turn (best-effort).
    # -------------------------------------------------
    if memory is not None:
        try:
            memory.record_assistant(result, request.staged_property_ids)
        except Exception:
            logger.exception("Failed to record assistant turn in memory.")

    return result
