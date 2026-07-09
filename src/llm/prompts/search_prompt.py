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

from src.llm.prompts.config import PromptConfig, DEFAULT_CONFIG
from src.llm.prompts.templates import Prompt, build_prompt
from src.llm.prompts.formatting import format_records, format_mapping


def build_search_prompt(
    user_query: str,
    results: list[dict],
    filters: dict | None = None,
    weights: dict | None = None,
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
        config     : Shared prompt configuration.

    Returns:
        Prompt: A built prompt (system + user text). No Claude call is made.
    """

    backend_data = "\n\n".join(
        [
            "Active search filters:\n"
            + format_mapping(filters or {}),
            "Preference / ranking weights:\n"
            + format_mapping(weights or {}),
            "Recommended properties (already ranked by the backend):\n"
            + format_records(results or [], item_label="Property"),
        ]
    )

    task_instructions = (
        "Explain, in natural language, why these properties were "
        "recommended for the user's query. Base your explanation only on "
        "the ranking scores, filters, preference weights and the backend's "
        "'why recommended' notes shown above. The backend has already "
        "searched and ranked these properties; do not re-rank them, add "
        "properties, or invent details that are not present."
    )

    expected_output = (
        "A short, friendly summary that highlights the top matches and "
        "clearly explains how they fit the user's request, followed by a "
        "brief note on the ranking rationale. If a detail is missing, say "
        "so rather than guessing."
    )

    return build_prompt(
        user_intent=f'The user searched: "{user_query}"',
        backend_data=backend_data,
        task_instructions=task_instructions,
        expected_output=expected_output,
        config=config,
    )
