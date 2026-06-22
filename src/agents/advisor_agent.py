# # ===============================
# # advisor_agent.py
# # ===============================

# import pandas as pd


# def run_advisor_agent(df: pd.DataFrame):
#     """
#     Generate advisor insights from existing
#     property analysis data.

#     Recommendation should come from
#     comparison_result, not from this agent.
#     """

#     print("✅ advisor_agent executed")

#     results = []

#     for _, row in df.iterrows():

#         positives = []
#         risks = []

#         # -------------------------
#         # POSITIVES
#         # -------------------------
#         if row.get("investment_rating") == "Excellent":
#             positives.append("Strong rental returns")

#         if row.get("growth_score", 0) >= 3:
#             positives.append("High future growth potential")

#         if row.get("risk_score", 0) <= 2:
#             positives.append("Low risk profile")

#         if row.get("analysis_flag") == "undervalued":
#             positives.append("Looks undervalued")

#         if row.get("negotiation_power") == "High":
#             positives.append("Strong negotiation opportunity")

#         # -------------------------
#         # RISKS
#         # -------------------------
#         if row.get("risk_score", 0) >= 6:
#             risks.append("High risk factors detected")

#         if row.get("analysis_flag") == "overpriced":
#             risks.append("Appears overpriced")

#         if row.get("growth_score", 0) == 0:
#             risks.append("Limited future growth signals")

#         if row.get("investment_rating") == "Low":
#             risks.append("Weak rental investment profile")

#         # -------------------------
#         # BUYER TYPE
#         # -------------------------
#         if row.get("investment_rating") == "Excellent":
#             suitable_for = "Rental Investor"

#         elif row.get("growth_score", 0) >= 3:
#             suitable_for = "Long-Term Investor"

#         else:
#             suitable_for = "End User"

#         results.append({

#             "id": row.get("id"),

#             "suitable_for": suitable_for,

#             "positives": " | ".join(positives),

#             "risks": " | ".join(risks)

#         })

#     return pd.DataFrame(results)

#=================================================================================================================================================================================

# import pandas as pd


# def run_advisor_agent(df: pd.DataFrame) -> pd.DataFrame:
#     """Collects both deep metrics for LLM context and surface-level summaries.

#     Includes concrete target indicators while parsing inputs defensively to
#     prevent NoneType comparison errors.
#     """
#     print("✅ advisor_agent executed")

#     results = []

#     for _, row in df.iterrows():
#         # -------------------------
#         # DEFENSIVE VARIABLE EXTRACT (Prevents NoneType issues)
#         # -------------------------
#         growth_score = row.get("growth_score") or 0
#         risk_score = row.get("risk_score") or 0
#         investment_rating = row.get("investment_rating") or ""
#         analysis_flag = row.get("analysis_flag") or ""
#         negotiation_power = row.get("negotiation_power") or ""

#         print(
#             f"""
#     ID={row.get('id')}
#     risk_score={risk_score}
#     analysis_flag={analysis_flag}
#     growth_score={growth_score}
#     investment_rating={investment_rating}

#     analysis_msg={row.get('analysis_msg')}
#     price_position={row.get('price_position')}
#     """
#         )

#         positives = []
#         risks = []

#         # -------------------------
#         # POSITIVES (UI Summary)
#         # -------------------------
#         if investment_rating == "Excellent":
#             positives.append("Strong rental returns")

#         if growth_score >= 3:
#             positives.append("High future growth potential")

#         if risk_score <= 2:
#             positives.append("Low risk profile")

#         if analysis_flag == "undervalued":
#             positives.append("Looks undervalued")

#         if negotiation_power == "High":
#             positives.append("Strong negotiation opportunity")

#         # -------------------------
#         # RISKS (UI Summary)
#         # -------------------------
#         if risk_score >= 6:
#             risks.append("High risk factors detected")

#         if analysis_flag == "overpriced":
#             risks.append("Appears overpriced")

#         if growth_score == 0:
#             risks.append("Limited future growth signals")

#         if investment_rating == "Low":
#             risks.append("Weak rental investment profile")

#         # ADD HERE
#         if not risks:
#             risks.append("No major risk factors detected")

#         # -------------------------
#         # BUYER TYPE (UI Summary)
#         # -------------------------
#         if investment_rating == "Excellent":
#             suitable_for = "Rental Investor"
#         elif growth_score >= 3:
#             suitable_for = "Long-Term Investor"
#         else:
#             suitable_for = "End User"

#         # -------------------------
#         # COMBINED DATASET
#         # -------------------------
#         results.append(
#             {
#                 "id": row.get("id"),
#                 # Valuation Layer
#                 "analysis_flag": row.get("analysis_flag"),
#                 "analysis_msg": row.get("analysis_msg"),
#                 "price_position": row.get("price_position"),
#                 # Risk Layer
#                 "risk_label": row.get("risk_label"),
#                 "risk_score": row.get("risk_score"),
#                 # Growth Layer
#                 "growth_label": row.get("growth_label"),
#                 "growth_reason": row.get("growth_reason"),
#                 "dev_summary": row.get("dev_summary"),
#                 # Rental Layer
#                 "investment_rating": row.get("investment_rating"),
#                 "rental_strategy": row.get("rental_strategy"),
#                 "rental_yield_percent": row.get("rental_yield_percent"),
#                 "demand_level": row.get("demand_level"),
#                 # Negotiation Layer
#                 "negotiation_power": row.get("negotiation_power"),
#                 "negotiation_score": row.get("negotiation_score"),
#                 "suggested_discount_percent": row.get("suggested_discount_percent"),
#                 "target_price": row.get("target_price"),
#                 # Summary Layer
#                 "positives": " | ".join(positives),
#                 "risks": " | ".join(risks),
#                 "suitable_for": suitable_for,
#             }
#         )

#     return pd.DataFrame(results)

#============================================================================================================================================================================================
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
            "dev_summary",

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