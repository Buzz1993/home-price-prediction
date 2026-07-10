# ===============================
# src/llm/tool_orchestrator.py
# ===============================
#
# Phase 15.8 — Intelligent Tool Orchestration (SELECTION only).
#
# This module lets Claude act as a ROUTER: given a user's natural-language
# message, it selects WHICH existing backend capability should handle it.
#
# What this module IS:
#   - A small registry of the EXISTING backend tools (name + description).
#   - A `select_tool()` helper that reuses the Phase 15.2 Prompt Builder
#     (build_orchestration_prompt) and the Phase 15.1 Claude Client
#     (ask_claude) to pick exactly one tool.
#
# What this module is NOT:
#   - It performs NO business logic. It does not search, rank, predict, value,
#     compare, recommend, generate reports or touch the tray. It only decides
#     which existing capability the caller should invoke.
#   - It does NOT execute the selected tool. Execution stays in the existing
#     backend (the API layer delegates to the existing services / MCP tools).
#
# Claude is OPTIONAL. If routing cannot be determined for any reason,
# `select_tool()` returns None and the caller falls back to the existing
# backend chat behaviour (chat_service.parse_intent_and_execute).

import re
import json
import logging
from dataclasses import dataclass

from src.llm.claude_client import ask_claude
from src.llm.prompts import ToolSpec, build_orchestration_prompt

logger = logging.getLogger(__name__)


# =====================================================
# AVAILABLE BACKEND TOOLS (existing capabilities only)
# =====================================================
#
# Every entry maps to an EXISTING backend capability. This registry only
# describes them for the router; it never implements them.

TOOLS: list[ToolSpec] = [
    ToolSpec(
        "search",
        "Find, browse, filter or show more properties by location, BHK, "
        "budget or amenities.",
    ),
    ToolSpec(
        "comparison",
        "Compare the staged properties and pick the best one "
        "(needs at least two staged).",
    ),
    ToolSpec(
        "prediction",
        "Predict / estimate the machine-learning price of the staged "
        "properties.",
    ),
    ToolSpec(
        "rental",
        "Estimate rental income or rental yield of the staged properties.",
    ),
    ToolSpec(
        "valuation",
        "Fair-market valuation — whether a staged property is over- or "
        "under-priced.",
    ),
    ToolSpec(
        "negotiation",
        "Negotiation strategy, target price or discount for the staged "
        "properties.",
    ),
    ToolSpec(
        "advisor",
        "Overall investment advice / should-I-buy verdict for the staged "
        "properties.",
    ),
    ToolSpec(
        "report",
        "Generate a written property report for the staged properties.",
    ),
    ToolSpec(
        "share_report",
        "Generate and send a property report to a phone number "
        "(WhatsApp / SMS).",
    ),
    ToolSpec(
        "saved",
        "Show the user's saved / bookmarked properties.",
    ),
    ToolSpec(
        "save_property",
        "Save / bookmark the staged property.",
    ),
    ToolSpec(
        "chat",
        "General real-estate question or anything that does not clearly map "
        "to another tool.",
    ),
]

# Fast lookup of valid tool names + the special 'clarify' outcome.
VALID_TOOL_NAMES: set[str] = {t.name for t in TOOLS}
CLARIFY = "clarify"


# =====================================================
# SELECTION RESULT
# =====================================================

@dataclass
class ToolSelection:
    """
    The router's decision.

    tool    : The selected tool name (one of TOOLS) or 'clarify'.
    reason  : Short internal reasoning (never shown to the user).
    message : A short follow-up question to ask the user (only when the tool
              is 'clarify').
    """

    tool: str
    reason: str | None = None
    message: str | None = None


# =====================================================
# ROUTER
# =====================================================

def select_tool(
    user_message: str,
    tray_ids: list[str] | None = None,
    memory: str | None = None,
) -> ToolSelection | None:
    """
    Ask Claude which EXISTING backend capability should handle the message.

    Args:
        user_message : The user's natural-language request.
        tray_ids     : Property ids currently staged in the evaluation tray.
        memory       : Optional session conversation-memory summary
                       (Phase 15.7), used only to resolve follow-up references.

    Returns:
        A ToolSelection when Claude confidently routes the request, or None
        when routing is unavailable/unparseable so the caller can fall back to
        the existing backend chat behaviour. This function never raises.
    """

    if not (user_message or "").strip():
        return None

    try:
        prompt = build_orchestration_prompt(
            user_message=user_message,
            tools=TOOLS,
            tray_ids=tray_ids,
            memory=memory,
        )

        # Routing is a small, deterministic classification task.
        response = ask_claude(
            prompt.user,
            system=prompt.system,
            temperature=0.0,
            max_tokens=200,
        )
    except Exception:
        # The router must never break the chat flow.
        logger.exception("Tool routing failed; falling back to backend chat.")
        return None

    if not response.success:
        logger.warning(
            "Tool routing unavailable (error_type=%s); falling back.",
            response.error_type,
        )
        return None

    return _parse_selection(response.text)


# =====================================================
# PARSING (robust, non-raising)
# =====================================================

def _parse_selection(text: str) -> ToolSelection | None:
    """
    Parse Claude's JSON routing decision into a validated ToolSelection.

    Returns None when the response cannot be parsed or names an unknown tool,
    so the caller falls back to the existing backend chat behaviour.
    """

    if not text:
        return None

    # Claude is instructed to return only JSON, but stay defensive and extract
    # the first JSON object if any stray text slips in.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        logger.warning("Tool routing returned no JSON object; falling back.")
        return None

    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        logger.warning("Tool routing returned invalid JSON; falling back.")
        return None

    if not isinstance(data, dict):
        return None

    tool = str(data.get("tool") or "").strip().lower()
    reason = data.get("reason")
    message = data.get("message")

    if tool == CLARIFY:
        return ToolSelection(
            tool=CLARIFY,
            reason=str(reason) if reason else None,
            message=str(message) if message else None,
        )

    if tool in VALID_TOOL_NAMES:
        return ToolSelection(
            tool=tool,
            reason=str(reason) if reason else None,
        )

    logger.warning("Tool routing chose an unknown tool '%s'; falling back.", tool)
    return None
