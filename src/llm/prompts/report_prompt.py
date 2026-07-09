# ===============================
# src/llm/prompts/report_prompt.py
# ===============================
#
# Prompt builder for REPORTS.
#
# Input: an existing backend-generated report (Markdown text produced by
# tools.create_property_report). This builder asks Claude to improve the
# readability of that report and summarize it, WITHOUT changing any of the
# facts, figures or conclusions the backend already produced.
#
# It performs no report generation or reasoning of its own.

from src.llm.prompts.config import PromptConfig, DEFAULT_CONFIG
from src.llm.prompts.templates import Prompt, build_prompt
from src.llm.prompts.formatting import format_value


def build_report_prompt(
    report: str,
    config: PromptConfig = DEFAULT_CONFIG,
) -> Prompt:
    """
    Build a prompt that asks Claude to improve a backend-generated report.

    Args:
        report : The existing report content (typically Markdown) generated
                 by the backend.
        config : Shared prompt configuration.

    Returns:
        Prompt: A built prompt (system + user text). No Claude call is made.
    """

    backend_data = (
        "Existing backend-generated report:\n"
        "-----\n"
        f"{format_value(report)}\n"
        "-----"
    )

    task_instructions = (
        "Improve the readability of the report above and provide a clear "
        "summary. You may reorganize, clarify wording and tighten structure, "
        "but you must preserve every fact, figure, price and conclusion "
        "exactly as written. The backend is responsible for the report's "
        "content; do not add new data, change any number, or introduce "
        "analysis that is not already in the report."
    )

    expected_output = (
        "A polished, well-structured version of the report (keeping Markdown "
        "formatting) with a short executive summary at the top. Do not alter "
        "any factual content. If information is missing, leave it out rather "
        "than inventing it."
    )

    return build_prompt(
        user_intent="The user wants a clearer, more readable report.",
        backend_data=backend_data,
        task_instructions=task_instructions,
        expected_output=expected_output,
        config=config,
    )
