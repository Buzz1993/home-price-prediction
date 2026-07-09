# ===============================
# src/llm/search_explanation.py
# ===============================
#
# Phase 15.3 — AI Search Explanation.
#
# Thin orchestration layer that turns an EXISTING backend search result into a
# natural-language explanation of WHY those properties were recommended.
#
# It performs NO business logic:
#   - It does not search, rank, filter or recommend.
#   - It never invents properties, prices or scores.
#   - It only reuses the Phase 15.2 Search Prompt Builder + Phase 15.1 Claude
#     Client to explain the structured results the backend already produced.
#
# Claude is OPTIONAL. If the explanation cannot be generated for any reason,
# this module returns None and the caller still returns the backend search
# results unchanged.

import logging

from src.llm.claude_client import ask_claude
from src.llm.prompts import build_search_prompt

logger = logging.getLogger(__name__)


def explain_search_results(
    user_query: str,
    results: list[dict],
    filters: dict | None = None,
    weights: dict | None = None,
) -> str | None:
    """
    Generate a conversational explanation for a backend search result.

    Args:
        user_query : The original user search query.
        results    : Ranked property records already produced by the backend
                     (chat_service.parse_intent_and_execute -> "content").
        filters    : Active search filters used by the backend (optional).
        weights    : Preference / hybrid ranking weights (optional).

    Returns:
        The explanation text, or None when Claude is unavailable / fails or
        there is nothing to explain. Returning None lets the caller keep the
        search results working without an explanation.
    """

    # Nothing to explain — do not spend a Claude call on an empty result set.
    if not results:
        return None

    # Reuse the Phase 15.2 builder; never build prompts inline here.
    prompt = build_search_prompt(
        user_query=user_query,
        results=results,
        filters=filters,
        weights=weights,
    )

    # Reuse the Phase 15.1 Claude client. It never raises: failures come back
    # as a structured, unsuccessful response.
    response = ask_claude(prompt.user, system=prompt.system)

    if not response.success:
        # Log and degrade gracefully; the caller keeps the search results.
        logger.warning(
            "Search explanation unavailable (error_type=%s).",
            response.error_type,
        )
        return None

    return response.text
