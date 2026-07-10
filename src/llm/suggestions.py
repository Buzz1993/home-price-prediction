# ===============================
# src/llm/suggestions.py
# ===============================
#
# Phase 15.11 — AI Suggestions.
#
# Thin orchestration layer that turns a completed chat turn into a short list
# of useful FOLLOW-UP ACTIONS the user can take next.
#
# It performs NO business logic:
#   - It does not search, rank, predict, value, compare, recommend or run any
#     analysis. It never invents properties, prices, scores or features.
#   - It only reuses the Phase 15.2 Suggestions Prompt Builder + Phase 15.1
#     Claude Client to recommend EXISTING EstateMind capabilities as next steps.
#
# Every suggestion is a short, natural-language action phrase. Selecting one in
# the frontend simply sends it back through the existing chat pipeline, where
# the Phase 15.8 tool orchestration routes it to the right EXISTING backend
# capability — so no new workflow or routing logic is introduced here.
#
# Claude is OPTIONAL. If suggestions cannot be generated for any reason, this
# module returns an empty list and the caller hides the suggestion section; the
# chat response is never affected.

import re
import json
import logging
from dataclasses import dataclass

from src.llm.claude_client import ask_claude
from src.llm.prompts import build_suggestions_prompt

logger = logging.getLogger(__name__)


# =====================================================
# EXISTING ESTATEMIND CAPABILITIES (next-action catalog)
# =====================================================
#
# Each Action is an EXISTING EstateMind capability, phrased as a short request
# the user could send. `min_tray` is how many staged (evaluation-tray)
# properties the capability needs, so we only ever suggest actions the user can
# actually run right now. This catalog only DESCRIBES existing features; it
# implements nothing.

@dataclass(frozen=True)
class Action:
    label: str        # the action phrase shown to the user and sent as a message
    min_tray: int = 0  # staged properties required to run it


# Tray-independent actions (always available).
_SEARCH_MORE = Action("Show me more matching properties", 0)
_REFINE_SEARCH = Action("Refine my search", 0)
_VIEW_SAVED = Action("View my saved properties", 0)

# Tray-dependent actions (need at least one / two staged properties).
_COMPARE = Action("Compare the staged properties", 2)
_PREDICT = Action("Predict the price", 1)
_RENTAL = Action("Estimate the rental income", 1)
_VALUATION = Action("Check if it is fairly priced", 1)
_NEGOTIATION = Action("Get a negotiation strategy", 1)
_ADVISOR = Action("Should I invest in it?", 1)
_REPORT = Action("Generate a full investment report", 1)
_SHARE_REPORT = Action("Share the report on WhatsApp", 1)
_SAVE = Action("Save this property for later", 1)


# Candidate next actions per backend response type, ordered by usefulness. The
# lists mirror the natural workflow (search -> compare/analyse -> report -> …).
NEXT_ACTIONS: dict[str, list[Action]] = {
    "search_results": [
        _COMPARE,
        _PREDICT,
        _RENTAL,
        _ADVISOR,
        _REPORT,
        _SAVE,
        _SEARCH_MORE,
    ],
    "comparison": [
        _REPORT,
        _SHARE_REPORT,
        _NEGOTIATION,
        _ADVISOR,
        _SAVE,
    ],
    "prediction": [
        _VALUATION,
        _RENTAL,
        _ADVISOR,
        _REPORT,
    ],
    "rental": [
        _PREDICT,
        _ADVISOR,
        _REPORT,
        _SAVE,
    ],
    "valuation": [
        _NEGOTIATION,
        _ADVISOR,
        _RENTAL,
        _REPORT,
    ],
    "negotiation": [
        _REPORT,
        _ADVISOR,
        _SHARE_REPORT,
    ],
    "advisor": [
        _REPORT,
        _NEGOTIATION,
        _RENTAL,
        _SAVE,
    ],
    # Reports, saved-property replies, clarifications and general chat.
    "text": [
        _COMPARE,
        _ADVISOR,
        _REPORT,
        _VIEW_SAVED,
        _REFINE_SEARCH,
    ],
}

# Human-readable label for the prompt describing what just happened.
_RESULT_CONTEXT: dict[str, str] = {
    "search_results": "ranked property search results",
    "comparison": "a comparison of the staged properties with a recommended winner",
    "prediction": "a machine-learning price prediction",
    "rental": "a rental income / yield analysis",
    "valuation": "a fair-market valuation",
    "negotiation": "a negotiation strategy",
    "advisor": "overall investment advice",
    "text": "a text response",
}

# Bounds — keep suggestions short and few.
MAX_SUGGESTIONS = 5
MAX_SUGGESTION_LEN = 80


# =====================================================
# SUGGESTION GENERATION
# =====================================================

def generate_suggestions(
    user_message: str,
    result: dict,
    staged_property_ids: list[str] | None = None,
    memory: str | None = None,
) -> list[str]:
    """
    Recommend useful next actions after a chat turn.

    Args:
        user_message        : The user's latest request.
        result              : The structured backend response envelope
                              (search_results / comparison / analysis / text).
        staged_property_ids : Property ids currently in the evaluation tray, so
                              only runnable actions are suggested.
        memory              : Optional session conversation-memory summary
                              (Phase 15.7). Context only.

    Returns:
        A list of short action phrases (3-5), or an empty list when suggestions
        are unavailable / fail. This function never raises.
    """

    if not isinstance(result, dict):
        return []

    rtype = result.get("type")

    # Nothing to act on for an empty search result — skip the Claude call.
    if rtype == "search_results" and not result.get("content"):
        return []

    candidates = NEXT_ACTIONS.get(rtype)
    if not candidates:
        return []

    staged_count = len(staged_property_ids or [])
    available = [a.label for a in candidates if a.min_tray <= staged_count]
    if not available:
        return []

    tray_state = (
        f"The user has {staged_count} property(ies) staged in their evaluation "
        "tray."
        if staged_count
        else "The user's evaluation tray is empty."
    )

    try:
        prompt = build_suggestions_prompt(
            user_message=user_message,
            result_context=_RESULT_CONTEXT.get(rtype, "a response"),
            available_actions=available,
            tray_state=tray_state,
            memory=memory,
        )

        # Suggestions are a small, low-creativity selection task.
        response = ask_claude(
            prompt.user,
            system=prompt.system,
            temperature=0.2,
            max_tokens=200,
        )
    except Exception:
        # Suggestions must never break the chat flow.
        logger.exception("Suggestion generation failed; returning none.")
        return []

    if not response.success:
        logger.warning(
            "Suggestions unavailable (error_type=%s).", response.error_type
        )
        return []

    return _parse_suggestions(response.text)


# =====================================================
# PARSING (robust, non-raising)
# =====================================================

def _parse_suggestions(text: str) -> list[str]:
    """
    Parse Claude's JSON suggestions into a clean, bounded list of strings.

    Accepts either {"suggestions": [...]} or a bare JSON array. Returns an empty
    list when the response cannot be parsed, so the caller hides the section.
    """

    if not text:
        return []

    # Claude is instructed to return only JSON; stay defensive and extract the
    # first JSON object or array if any stray text slips in.
    match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
    if not match:
        return []

    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return []

    if isinstance(data, dict):
        items = data.get("suggestions")
    else:
        items = data

    if not isinstance(items, list):
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        label = item.strip()
        if not label or len(label) > MAX_SUGGESTION_LEN:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(label)
        if len(cleaned) >= MAX_SUGGESTIONS:
            break

    return cleaned
