# ===============================
# src/llm/prompts/comparison_prompt.py
# ===============================
#
# Prompt builder for COMPARISON results.
#
# Input: the structured comparison response produced by the existing backend
# (tools.compare_properties), which returns:
#     { "winner": {...}, "rankings": [ {...}, ... ] }
# where each ranking entry includes id, overall_score, verdict and
# comparison_reason.
#
# This builder only formats that data and asks Claude to explain the
# comparison. It performs no ranking, scoring or reasoning.

from src.llm.prompts.config import PromptConfig, DEFAULT_CONFIG
from src.llm.prompts.templates import Prompt, build_prompt
from src.llm.prompts.formatting import format_record, format_records


def build_comparison_prompt(
    comparison: dict,
    config: PromptConfig = DEFAULT_CONFIG,
) -> Prompt:
    """
    Build a prompt that asks Claude to explain backend comparison results.

    Args:
        comparison : The structured comparison response from the backend,
                     containing a "winner" record and a "rankings" list.
        config     : Shared prompt configuration.

    Returns:
        Prompt: A built prompt (system + user text). No Claude call is made.
    """

    comparison = comparison or {}
    winner = comparison.get("winner") or {}
    rankings = comparison.get("rankings") or []

    backend_data = "\n\n".join(
        [
            "Backend-selected best property (winner):\n"
            + format_record(winner, indent="  "),
            "Full ranking produced by the backend:\n"
            + format_records(rankings, item_label="Rank"),
        ]
    )

    task_instructions = (
        "Explain the backend's property comparison in natural language. "
        "Describe why the winning property came out on top and summarize the "
        "relative strengths and weaknesses of each option using the scores "
        "and verdicts shown above. The backend has already scored and ranked "
        "these properties; do not change the winner, re-rank them, or invent "
        "comparison points that are not in the data."
    )

    expected_output = (
        "A clear comparison summary that names the recommended property, "
        "explains the reasoning behind the ranking, and briefly contrasts the "
        "options. If a value is missing, say so instead of guessing."
    )

    return build_prompt(
        user_intent="The user wants to compare the selected properties.",
        backend_data=backend_data,
        task_instructions=task_instructions,
        expected_output=expected_output,
        config=config,
    )
