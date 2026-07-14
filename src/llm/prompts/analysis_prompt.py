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

from src.llm.prompts.config import (
    PromptConfig,
    DEFAULT_CONFIG,
    PRICE_INTERPRETATION_RULES,
    CARD_FORMAT_RULES,
    MULTI_PROPERTY_RULES,
)
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

# What the final "Overall Recommendation" comparison should single out for
# each analysis type, using backend values only. Applied after the
# per-property sections when more than one property is analyzed.
ANALYSIS_COMPARISON_FOCUS = {
    "prediction": (
        "identify which property offers the best value versus its asking "
        "price"
    ),
    "rental": (
        "identify the highest rental income, the highest rental yield and "
        "the best overall rental opportunity"
    ),
    "risk": "identify the lowest-risk and the highest-risk property",
    "valuation": (
        "identify the most undervalued and the most overpriced property"
    ),
    "future": (
        "identify the strongest and the weakest future growth potential"
    ),
    "future_growth": (
        "identify the strongest and the weakest future growth potential"
    ),
    "negotiation": (
        "state each property's target price, suggested discount and "
        "negotiation opportunity, then conclude which property has the "
        "strongest negotiation leverage"
    ),
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

    key = str(analysis_type).lower().strip()
    label = ANALYSIS_LABELS.get(key, f"{str(analysis_type).title()} Analysis")

    records = analysis or []
    count = len(records)

    backend_data = (
        f"{label} results produced by the backend ({count} "
        f"propert{'y' if count == 1 else 'ies'}):\n"
        + format_records(records, item_label="Property")
    )

    task_instructions = (
        f"Explain the backend's {label} in clear, natural language for the "
        "user. Interpret the figures, flags, scores and messages exactly as "
        "provided. The backend has already computed this analysis; do not "
        "recompute any value, change any figure, or add analysis that is not "
        "present in the data above.\n\n"
        + (
            f"The backend returned 1 property, so your explanation must "
            "contain exactly 1 property section.\n\n"
            if count == 1
            else (
                f"The backend returned {count} properties, so your "
                f"explanation must contain exactly {count} property "
                "sections, followed by one Overall Recommendation "
                "comparison.\n\n"
            )
        )
        + MULTI_PROPERTY_RULES
        + "\n\n"
        + PRICE_INTERPRETATION_RULES
    )

    comparison_focus = ANALYSIS_COMPARISON_FOCUS.get(key)
    if count > 1 and comparison_focus:
        task_instructions += (
            f"\n\nIn the final Overall Recommendation, {comparison_focus}, "
            "explaining WHY using only the backend values above."
        )

    expected_output = (
        "A concise, easy-to-understand explanation of what the analysis "
        "means for the property, highlighting the key takeaways. If a field "
        "is missing or unavailable, state that clearly instead of assuming a "
        "value.\n\n"
        + CARD_FORMAT_RULES
    )

    return build_prompt(
        user_intent=f"The user wants to understand the {label} results.",
        backend_data=backend_data,
        task_instructions=task_instructions,
        expected_output=expected_output,
        config=config,
    )
