# ===============================
# src/llm/prompts/suggestions_prompt.py
# ===============================
#
# Prompt builder for AI SUGGESTIONS (Phase 15.11).
#
# This builder formats a prompt that asks Claude to recommend the most useful
# FOLLOW-UP ACTIONS after a chat turn. Claude only SUGGESTS next steps — it
# never executes them and never invents features:
#   - Every suggestion must come from the provided list of EXISTING EstateMind
#     capabilities (the same capabilities the backend already exposes).
#   - Suggestions are optional hints that continue the user's workflow.
#
# Like every Phase 15.2 builder, this module performs no AI reasoning and makes
# no API calls. It only assembles a plain-text prompt and reuses the shared
# APPLICATION CONTEXT and `Prompt` object.

from src.llm.prompts.config import APPLICATION_CONTEXT
from src.llm.prompts.templates import Prompt


# =====================================================
# SUGGESTIONS SYSTEM PROMPT
# =====================================================

SUGGESTIONS_SYSTEM_PROMPT = (
    "You are EstateMind's follow-up assistant.\n"
    "\n"
    "The EstateMind backend has already answered the user's request. Your ONLY "
    "job is to recommend a few useful NEXT ACTIONS the user could take next, "
    "chosen strictly from the list of existing EstateMind capabilities you are "
    "given. You never search, rank, predict, value, compare, recommend or run "
    "anything yourself, and you never invent features or actions.\n"
    "\n"
    "Respond with ONE JSON object and nothing else:\n"
    '  {"suggestions": ["<short action>", "<short action>", ...]}\n'
    "\n"
    "Rules:\n"
    "- Return 3 to 5 short, action-style suggestions (a few words each).\n"
    "- Every suggestion MUST correspond to one AVAILABLE ACTION below. You may "
    "lightly rephrase it to fit the conversation, but keep the same meaning.\n"
    "- Order them by how useful they are given the user's request, the "
    "conversation and the backend result.\n"
    "- Do not repeat the action the user just completed.\n"
    "- Never invent an action that is not in the AVAILABLE ACTIONS list.\n"
    "- Never add any text outside the JSON object."
)


# =====================================================
# SUGGESTIONS PROMPT BUILDER
# =====================================================

def build_suggestions_prompt(
    user_message: str,
    result_context: str,
    available_actions: list[str],
    tray_state: str | None = None,
    memory: str | None = None,
) -> Prompt:
    """
    Build a prompt asking Claude to suggest useful next actions.

    Args:
        user_message      : The user's latest natural-language request.
        result_context    : A short description of what the backend just
                            produced (e.g. "ranked search results",
                            "a property comparison"). Context only.
        available_actions : The EXISTING EstateMind capabilities Claude may
                            suggest from (already filtered for the current
                            context, e.g. tray state). Claude must not go
                            outside this list.
        tray_state        : Optional short note about the evaluation tray so
                            suggestions match what the user can act on.
        memory            : Optional session conversation-memory summary
                            (Phase 15.7). Context only — used to keep the
                            suggestions relevant and avoid repeating actions.

    Returns:
        Prompt: A built prompt (system + user). No Claude call is made here.
    """

    action_lines = "\n".join(f"- {action}" for action in available_actions)

    sections = [
        "## APPLICATION CONTEXT\n" + APPLICATION_CONTEXT,
        f'## USER REQUEST\n"{user_message}"',
        "## WHAT THE BACKEND JUST PRODUCED\n" + result_context,
        "## AVAILABLE ACTIONS (suggest ONLY from these)\n" + action_lines,
    ]

    if tray_state:
        sections.append("## EVALUATION TRAY STATE\n" + tray_state)

    if memory:
        sections.append(
            "## CONVERSATION MEMORY (context only — keep suggestions relevant, "
            "never invent facts)\n" + memory
        )

    sections.append(
        "## YOUR TASK\n"
        "Suggest the 3-5 most useful next actions from the AVAILABLE ACTIONS "
        "list, tailored to the user's request and the backend result. Keep each "
        "one short and actionable, and do not suggest the action the user just "
        "completed."
    )

    user = "\n\n".join(sections).strip()

    return Prompt(system=SUGGESTIONS_SYSTEM_PROMPT, user=user)
