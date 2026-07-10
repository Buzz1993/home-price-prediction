# ===============================
# src/llm/prompts/investment_prompt.py
# ===============================
#
# Prompt builder for the COMBINED INVESTMENT ADVISOR summary (Phase 15.6).
#
# Input: the structured results the existing backend analysis agents already
# produced for the SAME set of properties — the Investment Advisor result plus,
# when available, price prediction, rental analysis, valuation and negotiation.
# Each is returned by the backend as a `list[dict]` (one entry per property).
#
# This builder only assembles those existing results into ONE prompt and asks
# Claude to write a single, coherent investment summary that connects them. It
# performs no analysis, prediction, valuation, scoring or ranking of its own,
# and never invents a section for which the backend provided no data (e.g.
# Future Growth, which has no backend endpoint).

from src.llm.prompts.config import PromptConfig, DEFAULT_CONFIG
from src.llm.prompts.templates import Prompt, build_prompt
from src.llm.prompts.formatting import format_records
from src.llm.prompts.analysis_prompt import ANALYSIS_LABELS


# The backend analyses combined into the investment context, in the order they
# should appear in the prompt. "advisor" (the Investment Advisor result) is the
# anchor; the rest add supporting context when the backend provides them.
INVESTMENT_SECTIONS = (
    "advisor",
    "prediction",
    "valuation",
    "rental",
    "risk",
    "negotiation",
    "future",
)


def build_investment_prompt(
    analyses: dict,
    config: PromptConfig = DEFAULT_CONFIG,
) -> Prompt:
    """
    Build a prompt that asks Claude to explain the COMBINED backend analysis
    as a single investment summary.

    Args:
        analyses : A mapping of analysis type -> backend result (list of dicts),
                   e.g. { "advisor": [...], "prediction": [...], "rental": [...],
                   "valuation": [...], "negotiation": [...] }. Only the analyses
                   the backend actually produced are included; missing/empty
                   analyses are simply left out of the prompt.
        config   : Shared prompt configuration.

    Returns:
        Prompt: A built prompt (system + user text). No Claude call is made.
    """

    analyses = analyses or {}

    # One readable block per available backend analysis. An analysis that the
    # backend did not produce (missing key or empty list) is omitted entirely,
    # so Claude is never asked to summarize a section that has no data.
    blocks = []
    for key in INVESTMENT_SECTIONS:
        records = analyses.get(key)
        if not records:
            continue

        label = ANALYSIS_LABELS.get(key, f"{str(key).title()} Analysis")
        blocks.append(
            f"{label} results produced by the backend:\n"
            + format_records(records, item_label="Property")
        )

    backend_data = "\n\n".join(blocks) if blocks else format_records([])

    task_instructions = (
        "Combine the separate backend analyses above into ONE coherent, "
        "easy-to-understand investment summary for the selected property. "
        "Connect the individual results — the Investment Advisor verdict, and "
        "where available the price prediction, valuation, rental analysis, "
        "risk and negotiation — into a single narrative that explains what the "
        "backend concluded and why it fits together. Interpret the figures, "
        "flags, scores and verdicts exactly as provided. The backend has "
        "already performed every prediction, valuation, rental estimate, risk "
        "assessment and recommendation; do not recompute any value, change any "
        "figure, override the backend's recommendation, or add analysis "
        "(such as ROI, yield or future growth) that is not present above."
    )

    expected_output = (
        "A professional but conversational investment summary that draws only "
        "on the backend results. Where the data supports it, cover: an "
        "executive summary, investment strengths, investment risks, rental "
        "potential, valuation summary, price outlook, negotiation "
        "considerations, an overall recommendation and the key takeaways. "
        "Only include a section when the backend provided the relevant data — "
        "omit any section (for example future growth) that has no supporting "
        "data instead of inventing it. Always defer to the backend's "
        "Investment Advisor recommendation rather than forming a new one."
    )

    return build_prompt(
        user_intent=(
            "The user wants an overall investment recommendation and summary "
            "for the selected property."
        ),
        backend_data=backend_data,
        task_instructions=task_instructions,
        expected_output=expected_output,
        config=config,
    )
