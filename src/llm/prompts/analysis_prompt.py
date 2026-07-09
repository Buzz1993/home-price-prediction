# ===============================
# src/llm/prompts/analysis_prompt.py
# ===============================
#
# Prompt builder for PROPERTY ANALYSIS results.
#
# Input: the structured analysis produced by the existing backend analysis
# agents / tools, e.g. risk, rental, valuation, future growth, negotiation
# and price prediction. Each of these is returned by the backend as a
# `list[dict]` (one entry per property). This builder formats that data and
# asks Claude to explain it.
#
# It performs no analysis, valuation or reasoning of its own.

from src.llm.prompts.config import PromptConfig, DEFAULT_CONFIG
from src.llm.prompts.templates import Prompt, build_prompt
from src.llm.prompts.formatting import format_records


# Human-readable labels for the analysis types the backend already supports.
ANALYSIS_LABELS = {
    "risk": "Risk Analysis",
    "rental": "Rental Analysis",
    "valuation": "Valuation Analysis",
    "future": "Future Growth Analysis",
    "future_growth": "Future Growth Analysis",
    "negotiation": "Negotiation Strategy",
    "prediction": "Price Prediction",
    "analysis": "Price Analysis",
}


def build_analysis_prompt(
    analysis_type: str,
    analysis: list[dict],
    config: PromptConfig = DEFAULT_CONFIG,
) -> Prompt:
    """
    Build a prompt that asks Claude to explain a backend analysis result.

    Args:
        analysis_type : One of the supported analysis keys (risk, rental,
                        valuation, future, negotiation, prediction, ...).
                        Unknown keys are still handled gracefully.
        analysis      : The structured analysis records returned by the
                        backend (list of dicts, one per property).
        config        : Shared prompt configuration.

    Returns:
        Prompt: A built prompt (system + user text). No Claude call is made.
    """

    label = ANALYSIS_LABELS.get(
        str(analysis_type).lower().strip(),
        f"{str(analysis_type).title()} Analysis",
    )

    backend_data = (
        f"{label} results produced by the backend:\n"
        + format_records(analysis or [], item_label="Property")
    )

    task_instructions = (
        f"Explain the backend's {label} in clear, natural language for the "
        "user. Interpret the figures, flags, scores and messages exactly as "
        "provided. The backend has already computed this analysis; do not "
        "recompute any value, change any figure, or add analysis that is not "
        "present in the data above."
    )

    expected_output = (
        "A concise, easy-to-understand explanation of what the analysis "
        "means for each property, highlighting the key takeaways. If a field "
        "is missing or unavailable, state that clearly instead of assuming a "
        "value."
    )

    return build_prompt(
        user_intent=f"The user wants to understand the {label} results.",
        backend_data=backend_data,
        task_instructions=task_instructions,
        expected_output=expected_output,
        config=config,
    )
