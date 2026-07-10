# ===============================
# src/api/chat_api.py
# ===============================

import re
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.services.chat_service import parse_intent_and_execute
from src.llm.search_explanation import explain_search_results
from src.llm.conversation_memory import get_session_memory_store
from src.llm.tool_orchestrator import select_tool, CLARIFY
from src.mcp.tools.property_tools import (
    compare_properties,
    get_price_prediction,
    get_rental_analysis,
    get_valuation_analysis,
    get_negotiation_strategy,
    get_investment_advice,
    create_property_report,
    send_property_report,
)
from src.api.saved_api import (
    get_saved_properties,
    save_property,
    SavePropertyRequest,
)

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
# INTELLIGENT TOOL ORCHESTRATION (Phase 15.8)
# =====================================================
#
# Claude acts ONLY as a router: it selects which EXISTING backend capability
# should handle the message (src/llm/tool_orchestrator.select_tool). The
# dispatch below is thin delegation — it invokes the EXISTING backend services
# / MCP tools and wraps their unchanged output in the response envelope the
# frontend already renders. It performs no business logic of its own.
#
# When the router is unsure, unavailable or selects search/general chat, the
# dispatch returns None so the caller falls back to the existing backend chat
# behaviour (parse_intent_and_execute) exactly as before.

# Tray-based analyses -> (existing backend executor, minimum staged properties).
# The response `type` equals the tool name, matching the existing frontend
# renderers (comparison / prediction / rental / valuation / negotiation /
# advisor). These are the SAME functions the existing backend already uses.
TRAY_TOOL_EXECUTORS = {
    "comparison": (compare_properties, 2),
    "prediction": (get_price_prediction, 1),
    "rental": (get_rental_analysis, 1),
    "valuation": (get_valuation_analysis, 1),
    "negotiation": (get_negotiation_strategy, 1),
    "advisor": (get_investment_advice, 1),
}

# Tools that reuse the existing search / general-chat pipeline are handled by
# the fallback (parse_intent_and_execute), so the dispatch defers on them.
DELEGATE_TO_BACKEND_CHAT = {"search", "chat"}


def _tray_error(min_items: int) -> dict:
    """Friendly prompt to stage properties (mirrors the existing backend copy)."""
    plural = "ies" if min_items > 1 else "y"
    return {
        "type": "text",
        "content": (
            f"⚠️ Please add at least {min_items} propert{plural} to your "
            "evaluation tray first."
        ),
    }


def _extract_phone_number(text: str) -> str | None:
    """Pull a phone number out of the message for report sharing (data only)."""
    match = re.search(r"\+?\d[\d\s\-]{7,}\d", text or "")
    return re.sub(r"[\s\-]", "", match.group(0)) if match else None


def dispatch_selected_tool(selection, request: "ChatRequest"):
    """
    Route the router's decision to an EXISTING backend capability.

    Returns the structured response for the executed tool, or None to defer to
    the existing backend chat behaviour (search / general chat / unsure). All
    business logic stays in the existing backend; this only delegates and wraps
    the unchanged result.
    """

    if selection is None:
        return None

    tool = selection.tool
    tray = request.staged_property_ids or []

    # Search and general chat reuse the existing pipeline via the fallback.
    if tool in DELEGATE_TO_BACKEND_CHAT:
        return None

    # ----- Clarification (the router was not confident) -------------------
    if tool == CLARIFY:
        return {
            "type": "text",
            "content": selection.message
            or "Could you clarify what you'd like me to do with these properties?",
        }

    # ----- Tray-based analyses (existing MCP tools) -----------------------
    if tool in TRAY_TOOL_EXECUTORS:
        executor, min_items = TRAY_TOOL_EXECUTORS[tool]
        if len(tray) < min_items:
            return _tray_error(min_items)
        return {"type": tool, "content": execute(executor, tray)}

    # ----- Report generation (existing MCP tool) --------------------------
    if tool == "report":
        if not tray:
            return _tray_error(1)
        report = execute(create_property_report, tray)
        return {"type": "text", "content": report}

    # ----- Report sharing (existing MCP tools) ----------------------------
    if tool == "share_report":
        if not tray:
            return _tray_error(1)
        phone = _extract_phone_number(request.message)
        if not phone:
            return {
                "type": "text",
                "content": (
                    "Please share the phone number I should send the report "
                    "to (for example: +91 98765 43210)."
                ),
            }
        report = execute(create_property_report, tray)
        execute(send_property_report, phone, report)
        return {
            "type": "text",
            "content": f"✅ Your property report is on its way to {phone}.",
        }

    # ----- Saved properties (existing saved-property endpoints) -----------
    if tool == "saved":
        saved = execute(get_saved_properties)
        ids = saved.get("saved_properties", []) if isinstance(saved, dict) else []
        if not ids:
            return {
                "type": "text",
                "content": "You don't have any saved properties yet.",
            }
        return {
            "type": "text",
            "content": "Your saved properties: " + ", ".join(str(i) for i in ids),
        }

    if tool == "save_property":
        if not tray:
            return {
                "type": "text",
                "content": (
                    "Add a property to your evaluation tray (or use the "
                    "bookmark on a search result) so I know which one to save."
                ),
            }
        for pid in tray:
            execute(save_property, SavePropertyRequest(property_id=pid))
        return {
            "type": "text",
            "content": "✅ Saved: " + ", ".join(str(pid) for pid in tray),
        }

    # Unknown/unhandled tool -> defer to the existing backend chat behaviour.
    return None


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

    Intelligent tool orchestration (Phase 15.8): before falling back to the
    existing keyword workflow, Claude acts as a ROUTER and selects which
    EXISTING backend capability should handle the message (search, comparison,
    analysis, report, saved properties, ...). The selected tool is executed by
    the existing backend services / MCP tools — Claude never runs the tool or
    performs any business logic. When Claude is unsure, unavailable, or selects
    search / general chat, the endpoint falls back to the existing backend chat
    behaviour (parse_intent_and_execute) exactly as before.

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

    # -------------------------------------------------
    # Intelligent tool orchestration (Phase 15.8): let Claude route the message
    # to an EXISTING backend capability. Best-effort — any failure or an unsure
    # decision degrades to the existing backend chat behaviour below.
    # -------------------------------------------------
    result = None
    try:
        selection = select_tool(
            user_message=request.message,
            tray_ids=request.staged_property_ids,
            memory=memory_summary,
        )
        result = dispatch_selected_tool(selection, request)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Tool orchestration failed; using existing chat flow.")
        result = None

    # Fallback: the existing backend chat workflow (search, ranking,
    # recommendation, keyword-routed analyses and general chat). This also runs
    # whenever the router chose search / general chat or was unsure.
    if result is None:
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
