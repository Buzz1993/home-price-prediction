# ===============================
# chat_service.py
# ===============================

def build_context(
    recs=None,
    comparison_result=None,
    comparison_raw=None,
    last_explanation=None
):
    """
    Build compact property context for LLM.

    Priority:
    1. Input property
    2. Comparison raw (full enriched property data)
    3. Comparison result (summary scores/verdict)
    4. Explanation
    """

    sections = []

    # =====================================
    # INPUT PROPERTY
    # =====================================
    if recs and "input" in recs:

        input_df = recs["input"]

        if not input_df.empty:

            cols = [
                c for c in [
                    "id",
                    "project_name",
                    "location",
                    "price",
                    "area",
                    "bhk_type"
                ]
                if c in input_df.columns
            ]

            sections.append(
                "INPUT PROPERTY:\n"
                + input_df[cols].to_string(index=False)
            )

    # =====================================
    # ENRICHED PROPERTY DATA
    # =====================================
    if (
        comparison_raw is not None
        and not comparison_raw.empty
    ):

        important_cols = [

            # identity
            "id",
            "project_name",
            "location",

            # recommendation
            "why_recommended",
            "hybrid_score",

            # valuation
            "analysis_msg",

            # risk
            "risk_label",
            "risk_score",

            # growth
            "growth_label",
            "growth_reason",

            # rental
            "monthly_rent_estimate",
            "rental_yield_percent",
            "investment_rating",

            # negotiation
            "negotiation_power",
            "suggested_discount_percent",
            "target_price",
            "price_position",

            # development
            "dev_summary"
        ]

        cols = [
            c
            for c in important_cols
            if c in comparison_raw.columns
        ]

        sections.append(
            "PROPERTY DATA:\n"
            + comparison_raw[cols].to_string(index=False)
        )

    # =====================================
    # COMPARISON SUMMARY
    # =====================================
    if (
        comparison_result is not None
        and not comparison_result.empty
    ):

        cols = [
            c for c in [
                "id",
                "overall_score",
                "verdict",
                "comparison_reason"
            ]
            if c in comparison_result.columns
        ]

        sections.append(
            "COMPARISON SUMMARY:\n"
            + comparison_result[cols].to_string(index=False)
        )

    # =====================================
    # EXPLANATION
    # =====================================
    if last_explanation:

        sections.append(
            "COMPARISON INSIGHTS:\n"
            + str(last_explanation)
        )

    if not sections:
        return "No property data available."

    return "\n\n".join(sections)