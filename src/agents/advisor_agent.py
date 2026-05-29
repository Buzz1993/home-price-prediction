# ===============================
# advisor_agent.py
# ===============================

import pandas as pd


def run_advisor_agent(df: pd.DataFrame):
    """
    Generate advisor insights from existing
    property analysis data.

    Recommendation should come from
    comparison_result, not from this agent.
    """

    print("✅ advisor_agent executed")

    results = []

    for _, row in df.iterrows():

        positives = []
        risks = []

        # -------------------------
        # POSITIVES
        # -------------------------
        if row.get("investment_rating") == "Excellent":
            positives.append("Strong rental returns")

        if row.get("growth_score", 0) >= 3:
            positives.append("High future growth potential")

        if row.get("risk_score", 0) <= 2:
            positives.append("Low risk profile")

        if row.get("analysis_flag") == "undervalued":
            positives.append("Looks undervalued")

        if row.get("negotiation_power") == "High":
            positives.append("Strong negotiation opportunity")

        # -------------------------
        # RISKS
        # -------------------------
        if row.get("risk_score", 0) >= 6:
            risks.append("High risk factors detected")

        if row.get("analysis_flag") == "overpriced":
            risks.append("Appears overpriced")

        if row.get("growth_score", 0) == 0:
            risks.append("Limited future growth signals")

        if row.get("investment_rating") == "Low":
            risks.append("Weak rental investment profile")

        # -------------------------
        # BUYER TYPE
        # -------------------------
        if row.get("investment_rating") == "Excellent":
            suitable_for = "Rental Investor"

        elif row.get("growth_score", 0) >= 3:
            suitable_for = "Long-Term Investor"

        else:
            suitable_for = "End User"

        results.append({

            "id": row.get("id"),

            "suitable_for": suitable_for,

            "positives": " | ".join(positives),

            "risks": " | ".join(risks)

        })

    return pd.DataFrame(results)