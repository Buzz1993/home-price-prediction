# ===============================
# src/llm/prompts/orchestration_prompt.py
# ===============================
#
# Prompt builder for INTELLIGENT TOOL ORCHESTRATION (Phase 15.8).
#
# This builder formats a routing prompt that asks Claude to decide WHICH
# existing backend capability should handle the user's natural-language
# request. Claude acts ONLY as a router:
#   - It never searches, ranks, predicts, values, compares or recommends.
#   - It only SELECTS one existing backend tool by name.
#
# Like every Phase 15.2 builder, this module performs no AI reasoning and
# makes no API calls. It only assembles a plain-text prompt and reuses the
# shared `Prompt` object and the shared APPLICATION CONTEXT.

from dataclasses import dataclass

from src.llm.prompts.config import APPLICATION_CONTEXT
from src.llm.prompts.templates import Prompt


# =====================================================
# TOOL DESCRIPTOR
# =====================================================

@dataclass(frozen=True)
class ToolSpec:
    """
    A routable, EXISTING backend capability.

    `name`        : The tool identifier Claude must return.
    `description` : A short natural-language description used to help Claude
                    pick the right tool. This is documentation only — it never
                    changes what the backend does.
    """

    name: str
    description: str


# =====================================================
# ROUTER SYSTEM PROMPT
# =====================================================

ROUTER_SYSTEM_PROMPT = (
    "You are EstateMind's tool router.\n"
    "\n"
    "Your ONLY job is to read the user's message and decide which EXISTING "
    "backend capability should handle it. You never search, rank, predict, "
    "value, compare, recommend, generate reports or answer the request "
    "yourself — the EstateMind backend performs all of that. You only choose "
    "the single correct tool.\n"
    "\n"
    "Respond with ONE JSON object and nothing else:\n"
    '  {"tool": "<one tool name from the list>", "reason": "<short reason>"}\n'
    "\n"
    "If the request is genuinely ambiguous between tools, instead return:\n"
    '  {"tool": "clarify", "message": "<one short follow-up question>"}\n'
    "\n"
    "Never invent tool names. Never add any text outside the JSON object."
)


# =====================================================
# ORCHESTRATION PROMPT BUILDER
# =====================================================

def build_orchestration_prompt(
    user_message: str,
    tools: list[ToolSpec],
    tray_ids: list[str] | None = None,
    memory: str | None = None,
) -> Prompt:
    """
    Build a routing prompt asking Claude to select one backend tool.

    Args:
        user_message : The user's natural-language request.
        tools        : Available EXISTING backend capabilities (ToolSpec list).
        tray_ids     : Property ids currently in the evaluation tray. Some
                       tools (comparison, prediction, rental, ...) act on these
                       staged properties, so the router needs to know they
                       exist.
        memory       : Optional session conversation-memory summary
                       (Phase 15.7). Context only — used to resolve follow-up
                       references (e.g. "compare those", "the previous ones").

    Returns:
        Prompt: A built prompt (system + user). No Claude call is made here.
    """

    tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in tools)

    tray_ids = tray_ids or []
    if tray_ids:
        tray_block = (
            f"The user has {len(tray_ids)} property(ies) staged in their "
            f"evaluation tray: {', '.join(str(pid) for pid in tray_ids)}."
        )
    else:
        tray_block = "The user's evaluation tray is empty."

    sections = [
        "## APPLICATION CONTEXT\n" + APPLICATION_CONTEXT,
        "## AVAILABLE BACKEND TOOLS\n" + tool_lines,
        "## EVALUATION TRAY STATE\n" + tray_block,
    ]

    if memory:
        sections.append(
            "## CONVERSATION MEMORY (context only — resolve follow-up "
            "references, never invent facts)\n" + memory
        )

    sections.append(f'## USER MESSAGE\n"{user_message}"')

    sections.append(
        "## ROUTING RULES\n"
        "- Choose exactly ONE tool that best satisfies the request.\n"
        "- Use 'search' when the user wants to find, browse, filter or see "
        "more properties (by location, BHK, budget, amenities).\n"
        "- Tools that analyse staged properties (comparison, prediction, "
        "rental, valuation, negotiation, advisor, report, share_report, "
        "save_property) operate on the evaluation tray. Prefer them only when "
        "the user is clearly asking about their staged/selected properties.\n"
        "- 'comparison' needs at least two staged properties.\n"
        "- Use 'saved' to view bookmarked properties and 'save_property' to "
        "bookmark a staged property.\n"
        "- Use 'chat' for general real-estate questions or anything that does "
        "not clearly map to another tool.\n"
        "- When unsure, prefer 'chat' rather than guessing an analysis tool."
    )

    user = "\n\n".join(sections).strip()

    return Prompt(system=ROUTER_SYSTEM_PROMPT, user=user)
