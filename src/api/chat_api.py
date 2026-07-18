# ===============================
# src/api/chat_api.py
# ===============================

import re
import json
import math
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.services.chat_service import parse_intent_and_execute
from src.llm.search_explanation import (
    explain_search_results,
    stream_search_explanation,
)
from src.llm.conversation_memory import get_session_memory_store
from src.llm.tool_orchestrator import select_tool, CLARIFY
from src.llm.suggestions import generate_suggestions
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
    # Optional client-held search context (Phase 18.10). The frontend echoes
    # back the `current_query_state` of the LAST search response in the active
    # conversation ({"active_filters": ..., "chat_preference_weights": ...}).
    # It is used ONLY to re-seed the session's follow-up state when the
    # in-memory session no longer has it (server restart / TTL expiry), so
    # "show me more"-style follow-ups keep restoring the previous filters
    # exactly like the Streamlit st.session_state flow. Optional and additive —
    # existing clients and live sessions are unaffected.
    last_query_state: dict | None = Field(default=None)


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
# AI SUGGESTIONS (Phase 15.11)
# =====================================================

def attach_suggestions(result, user_message: str, tray_ids, memory: str | None = None):
    """
    Attach optional follow-up SUGGESTIONS to a completed chat response.

    The backend response is returned UNCHANGED — this only adds an optional
    `suggestions` list of short next-action phrases (Phase 15.11) drawn from the
    EXISTING EstateMind capabilities. Selecting one in the frontend re-sends it
    through the existing chat pipeline (Phase 15.8 tool orchestration), so no new
    workflow or routing logic is introduced.

    Claude is optional: if suggestions cannot be generated the response is
    returned as-is (the suggestion section is simply hidden). `memory` is the
    Phase 15.7 session summary passed through as context only.
    """

    if not isinstance(result, dict):
        return result

    try:
        suggestions = generate_suggestions(
            user_message=user_message,
            result=result,
            staged_property_ids=tray_ids,
            memory=memory,
        )
    except Exception:
        # Never let the suggestion layer break a working chat response.
        logger.exception("Suggestion step failed; returning response without it.")
        suggestions = []

    if suggestions:
        result["suggestions"] = suggestions

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
# SHARED CHAT WORKFLOW
# =====================================================
#
# The business logic shared by the JSON endpoint (POST /chat) and the streaming
# endpoint (POST /chat/stream). It produces the structured backend result
# WITHOUT the optional Claude search explanation and WITHOUT recording the
# assistant turn — those steps differ between the two endpoints (the streaming
# endpoint streams the explanation token-by-token). No business logic lives
# here; it only orchestrates the EXISTING backend pipeline, exactly as before.


def _run_chat_workflow(request: "ChatRequest"):
    """
    Run the existing chat pipeline and return (result, memory, memory_summary).

    result         : the structured backend response envelope (search results,
                     comparison, analysis, text, ...), before any AI explanation.
    memory         : the session ConversationMemory (or None if unavailable).
    memory_summary : compact memory context passed to Claude (or None).
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
    # Follow-up context re-hydration (Phase 18.10). The existing backend
    # follow-up logic (chat_service.is_followup_query + last_search_filters)
    # reads the session_state dict. In Streamlit that dict is st.session_state
    # and always survives with the visible conversation; over HTTP it lives in
    # the in-memory session store, which does NOT survive a server restart or
    # TTL expiry — while the frontend conversation (localStorage) does. When
    # that happens, "show me more such properties" finds no filters to restore
    # and falls into the generic-chat fallback instead of running the search.
    #
    # Fix: if the session has no recorded search yet but the client sent the
    # last search's own `current_query_state` back, seed the SAME keys the
    # existing workflow already reads. Pure state restoration at the API
    # boundary — no search, ranking or follow-up logic is changed, and live
    # sessions (which already have the keys) are untouched.
    # -------------------------------------------------
    if (
        session_state is not None
        and not session_state.get("last_search_filters")
        and isinstance(request.last_query_state, dict)
    ):
        filters = request.last_query_state.get("active_filters")
        weights = request.last_query_state.get("chat_preference_weights")
        if isinstance(filters, dict) and any(filters.values()):
            session_state["last_search_filters"] = filters
            if isinstance(weights, dict) and weights:
                session_state["last_search_weights"] = weights

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
        # TEMP LOG (routing debug — remove after verification)
        print("\n===== API ROUTING DEBUG =====")
        print(f"INCOMING MESSAGE        : {request.message!r}")
        print(f"ROUTER TOOL SELECTION   : {selection.tool if selection else None}")
        print(f"ROUTER REASON           : {selection.reason if selection else None}")
        print(f"HAS last_search_filters : {bool(session_state and session_state.get('last_search_filters'))}")
        print("=============================\n")
        result = dispatch_selected_tool(selection, request)
        if result is not None:
            print(f"API ROUTING DEBUG: handled by router branch type={result.get('type')}")  # TEMP LOG
        else:
            print("API ROUTING DEBUG: deferring to parse_intent_and_execute fallback")  # TEMP LOG
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

    return result, memory, memory_summary


def _record_assistant(memory, result, tray_ids):
    """Record the assistant turn in session memory (best-effort)."""
    if memory is None:
        return
    try:
        memory.record_assistant(result, tray_ids)
    except Exception:
        logger.exception("Failed to record assistant turn in memory.")


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

    result, memory, memory_summary = _run_chat_workflow(request)

    result = attach_search_explanation(result, request.message, memory=memory_summary)

    result = attach_suggestions(
        result,
        request.message,
        request.staged_property_ids,
        memory=memory_summary,
    )

    _record_assistant(memory, result, request.staged_property_ids)

    return result


# =====================================================
# CHAT STREAMING (Phase 15.9)
# =====================================================
#
# Streaming is a DELIVERY enhancement only. The backend business logic is
# identical to POST /chat (same _run_chat_workflow) — the sole difference is
# that Claude's natural-language SEARCH explanation is streamed token-by-token
# over Server-Sent Events instead of being returned in one shot. Every other
# response type (comparison / analysis / text) is structured or backend-produced
# text with no Claude tokens to stream, so it is delivered as a single `done`
# event after a brief `thinking` phase.
#
# The existing POST /chat JSON contract is untouched; this endpoint is purely
# additive and shares the same workflow, so no chat logic is duplicated.

# Only search results carry a streamable Claude explanation.
STREAMABLE_TYPE = "search_results"


def _json_safe(value):
    """
    Convert non-finite floats (NaN/Infinity) to null so backend DataFrame
    records serialize cleanly over SSE. Mirrors SafeJSONResponse in
    src/api/main.py (kept local to avoid importing the app at module load).
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _sse(payload: dict) -> str:
    """Serialize one Server-Sent Event (a single `data:` JSON line)."""
    return f"data: {json.dumps(_json_safe(payload))}\n\n"


def _stream_chat_events(result, request: "ChatRequest", memory, memory_summary):
    """
    Yield the SSE event stream for a chat response.

    Event shapes (each a `data: {json}` line):
      {"type": "thinking"}                     -> backend finished; response next
      {"type": "delta", "text": "..."}         -> incremental explanation tokens
      {"type": "done", "response": {...}}       -> full ChatResponse envelope
      {"type": "error", "message": "...",       -> the Claude stream failed;
       "recoverable": true}                        a `done` still follows with
                                                    any partial explanation.
    """

    # Signal the frontend to switch from the thinking indicator to the response.
    yield _sse({"type": "thinking"})

    # Stream Claude's explanation only for non-empty search results; everything
    # else is delivered directly as the final `done` payload.
    if (
        isinstance(result, dict)
        and result.get("type") == STREAMABLE_TYPE
        and result.get("content")
    ):
        query_state = result.get("current_query_state") or {}
        collected: list[str] = []
        try:
            for event in stream_search_explanation(
                user_query=request.message,
                results=result.get("content") or [],
                filters=query_state.get("active_filters"),
                weights=query_state.get("chat_preference_weights"),
                memory=memory_summary,
            ):
                if event.type == "delta":
                    collected.append(event.text)
                    yield _sse({"type": "delta", "text": event.text})
                elif event.type == "error":
                    # Degrade gracefully (Phase 15.3): keep any partial text and
                    # let the search results render without a full explanation.
                    logger.warning(
                        "Streamed search explanation failed (error_type=%s).",
                        event.error_type,
                    )
                    yield _sse(
                        {
                            "type": "error",
                            "message": event.error
                            or "The AI explanation was interrupted.",
                            "recoverable": True,
                        }
                    )
                # 'done' events carry the full text; we already accumulated the
                # deltas, so nothing extra is needed here.
        except Exception:
            # Never let the explanation layer break a working search response.
            logger.exception(
                "Search explanation stream failed; returning results without it."
            )

        explanation = "".join(collected).strip()
        if explanation:
            result["ai_explanation"] = explanation

    # Attach follow-up suggestions AFTER the streamed explanation is complete
    # (Phase 15.11). Suggestions are not streamed token-by-token — they travel in
    # the single `done` payload below, so they appear only once the response is
    # finished. Best-effort: a failure simply omits the suggestion section.
    result = attach_suggestions(
        result,
        request.message,
        request.staged_property_ids,
        memory=memory_summary,
    )

    # Final structured payload — identical envelope to POST /chat.
    yield _sse({"type": "done", "response": result})

    # Record the assistant turn once the response is fully delivered.
    _record_assistant(memory, result, request.staged_property_ids)


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """
    Streaming variant of POST /chat (Phase 15.9).

    Runs the SAME backend workflow as POST /chat and streams the result over
    Server-Sent Events: Claude's search explanation arrives token-by-token,
    followed by the full structured response envelope. All business logic,
    AI reasoning, search / analysis / comparison / report workflows and the
    POST /chat JSON contract are unchanged — only the delivery differs.

    The backend workflow runs BEFORE streaming begins, so any backend error
    surfaces as a normal HTTP error (the client can retry) rather than mid-stream.
    """

    # Run business logic first so failures return a proper HTTP status.
    result, memory, memory_summary = _run_chat_workflow(request)

    return StreamingResponse(
        _stream_chat_events(result, request, memory, memory_summary),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering for live tokens
        },
    )
