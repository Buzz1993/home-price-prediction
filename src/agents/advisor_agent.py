# ===============================
# advisor_agent.py
# ===============================

import pandas as pd

def run_advisor_agent(df: pd.DataFrame) -> pd.DataFrame:

    print("✅ advisor_agent executed")

    return df[
        [
            "id",

            # valuation
            "analysis_flag",
            "analysis_msg",
            "price_position",

            # risk
            "risk_label",
            "risk_score",

            # growth
            "growth_label",
            "growth_reason",


            # rental
            "investment_rating",
            "rental_strategy",
            "rental_yield_percent",
            "demand_level",

            # negotiation
            "negotiation_power",
            "negotiation_score",
            "suggested_discount_percent",
            "target_price",
        ]
    ].copy()