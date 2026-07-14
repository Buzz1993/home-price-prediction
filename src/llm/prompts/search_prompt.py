# ===============================
# src/llm/prompts/search_prompt.py
# ===============================
#
# Prompt builder for SEARCH results.
#
# Input: the structured search response produced by the existing backend
# (chat_service.parse_intent_and_execute / tools.search_properties), i.e.
# a list of recommended property records plus the query state (filters,
# preference weights). This builder only formats that data into a prompt
# asking Claude to explain WHY those properties were recommended.
#
# It performs no search, ranking or reasoning.

from src.llm.prompts.config import PromptConfig, DEFAULT_CONFIG, CARD_FORMAT_RULES
from src.llm.prompts.templates import Prompt, build_prompt
from src.llm.prompts.formatting import format_records, format_mapping


def build_search_prompt(
    user_query: str,
    results: list[dict],
    filters: dict | None = None,
    weights: dict | None = None,
    memory: str | None = None,
    config: PromptConfig = DEFAULT_CONFIG,
) -> Prompt:
    """
    Build a prompt that asks Claude to explain backend search results.

    Args:
        user_query : The original user search query.
        results    : Recommended property records returned by the backend
                     (each may include price, location, bhk_type,
                     search_score, why_recommended, ...).
        filters    : Active search filters used by the backend (optional).
        weights    : Preference / hybrid ranking weights (optional).
        memory     : Optional session conversation-memory summary (Phase 15.7).
                     Context only — it helps Claude resolve follow-up references
                     and is never treated as backend data to act on.
        config     : Shared prompt configuration.

    Returns:
        Prompt: A built prompt (system + user text). No Claude call is made.
    """

    data_sections = []

    # Conversation memory (Phase 15.7) is prepended as CONTEXT ONLY so Claude
    # can understand follow-up references without treating it as new facts.
    if memory:
        data_sections.append(
            "Conversation memory (context only — do not treat as backend "
            "results or invent from it):\n" + memory
        )

    data_sections.extend(
        [
            "Active search filters:\n"
            + format_mapping(filters or {}),
            "Preference / ranking weights:\n"
            + format_mapping(weights or {}),
            "Recommended properties (already ranked by the backend):\n"
            + format_records(results or [], item_label="Property"),
        ]
    )

    backend_data = "\n\n".join(data_sections)

    task_instructions = (
        "Explain, in natural language, why these properties were "
        "recommended for the user's query. Base your explanation only on "
        "the ranking scores, filters, preference weights and the backend's "
        "'why recommended' notes shown above. The backend has already "
        "searched and ranked these properties; do not re-rank them, add "
        "properties, or invent details that are not present."
    )

    if memory:
        task_instructions += (
            " Use the conversation memory only to keep continuity and resolve "
            "follow-up references (e.g. 'the cheaper ones', 'the previous "
            "property'); never invent remembered facts or present memory as "
            "backend data."
        )

    expected_output = (
        "A short, friendly summary that highlights the top matches and "
        "clearly explains how they fit the user's request, followed by a "
        "brief note on the ranking rationale. If a detail is missing, say "
        "so rather than guessing.\n\n"
        + CARD_FORMAT_RULES
    )

    return build_prompt(
        user_intent=f'The user searched: "{user_query}"',
        backend_data=backend_data,
        task_instructions=task_instructions,
        expected_output=expected_output,
        config=config,
    )
