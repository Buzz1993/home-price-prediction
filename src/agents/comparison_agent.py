# # ===============================
# # comparison_agent.py
# # ===============================
# import numpy as np
# import pandas as pd

# def normalize(series):
#     if series.max() == series.min():
#         return pd.Series([0.5] * len(series), index=series.index)
#     return (series - series.min()) / (series.max() - series.min())


# def run_comparison_agent(df):
#     """
#     Compare selected properties using:
#     - price_score
#     - risk_score
#     - growth_score

#     Calculates:
#     - overall_score
#     - verdict
#     - short comparison_reason

#     All these values are stored inside
#     the comparison result dataframe.

#     Returns final comparison dataframe.
#     """
#     print("✅ comparison_agent executed")
#     df = df.copy()
#     print("comparison_agent input df columns:", df.columns)

#     # -----------------------------
#     # PRICE SCORE (lower is better)
#     # -----------------------------
#     df["price_score"] = 1 - normalize(df["price"])

#     # -----------------------------
#     # RISK SCORE (lower is better)
#     # -----------------------------
#     if "risk_score" in df.columns:
#         df["risk_score_norm"] = 1 - normalize(df["risk_score"])
#     else:
#         df["risk_score_norm"] = 0.5

#     # -----------------------------
#     # GROWTH SCORE (higher is better)
#     # -----------------------------
#     if "growth_score" in df.columns:
#         df["growth_score_norm"] = normalize(df["growth_score"])
#     else:
#         df["growth_score_norm"] = 0.5

#     # -----------------------------
#     # FINAL SCORE
#     # -----------------------------
#     df["overall_score"] = (
#         0.4 * df["price_score"] +
#         0.3 * df["risk_score_norm"] +
#         0.3 * df["growth_score_norm"]
#     )

#     # -----------------------------
#     # VERDICT
#     # -----------------------------
#     def verdict(row):
#         if row["overall_score"] > 0.7:
#             return "🏆 Best Value"
#         elif row["risk_score"] >= 6:
#             return "⚠️ Risky"
#         elif row["price_score"] < 0.3:
#             return "💸 Expensive"
#         else:
#             return "👍 Balanced"

#     df["verdict"] = df.apply(verdict, axis=1)

#     def explain(row):
#         reasons = []

#         if row["price_score"] > 0.6:
#             reasons.append("better price")

#         if row["risk_score"] <= 2:
#             reasons.append("low risk")

#         if row["growth_score"] >= 2:
#             reasons.append("good future growth")

#         return ", ".join(reasons)

#     df["comparison_reason"] = df.apply(explain, axis=1)

#     return df[[
#         "id",
#         "price",
#         "risk_score",
#         "growth_score",
#         "overall_score",
#         "verdict",
#         "comparison_reason"   
#     ]]


#=============================================================================================


# # ===============================
# # comparison_agent.py
# # ===============================
# import numpy as np
# import pandas as pd

# def normalize(series):
#     if series.max() == series.min():
#         return pd.Series([0.5] * len(series), index=series.index)
#     return (series - series.min()) / (series.max() - series.min())


# def run_comparison_agent(df):
#     """
#     Compare selected properties using:
#     - price_score
#     - risk_score
#     - growth_score

#     Calculates:
#     - overall_score
#     - verdict
#     - short comparison_reason

#     All these values are stored inside
#     the comparison result dataframe.

#     Returns final comparison dataframe.
#     """
#     print("✅ comparison_agent executed")
#     df = df.copy()
#     #print("comparison_agent input df columns:", df.columns)

#     # -----------------------------
#     # PRICE SCORE (lower is better)
#     # -----------------------------
#     df["comparison_price_score"] = 1 - normalize(df["price"])

#     # -----------------------------
#     # RISK SCORE (lower is better)
#     # -----------------------------
#     if "risk_score" in df.columns:
#         df["risk_score_norm"] = 1 - normalize(df["risk_score"])
#     else:
#         df["risk_score_norm"] = 0.5

#     # -----------------------------
#     # GROWTH SCORE (higher is better)
#     # -----------------------------
#     if "growth_score" in df.columns:
#         df["growth_score_norm"] = normalize(df["growth_score"])
#     else:
#         df["growth_score_norm"] = 0.5

#     # -----------------------------
#     # HYBRID SCORE
#     # -----------------------------
#     if "hybrid_score" in df.columns:
#         df["hybrid_score_norm"] = normalize(df["hybrid_score"])
#     else:
#         df["hybrid_score_norm"] = 0.5


#     # -----------------------------
#     # LOCALITY SCORE
#     # -----------------------------
#     if "locality_rating" in df.columns:
#         df["locality_score_norm"] = normalize(df["locality_rating"])
#     else:
#         df["locality_score_norm"] = 0.5


#     # -----------------------------
#     # RENTAL YIELD SCORE
#     # -----------------------------
#     if "rental_yield_percent" in df.columns:

#         rental_clean = (
#             df["rental_yield_percent"]
#             .astype(str)
#             .str.replace("%", "")
#             .astype(float)
#         )

#         df["rental_yield_norm"] = normalize(rental_clean)

#     else:
#         df["rental_yield_norm"] = 0.5

#     # -----------------------------
#     # FINAL SCORE
#     # -----------------------------
#     df["overall_score"] = (
#         0.20 * df["comparison_price_score"] +
#         0.20 * df["risk_score_norm"] +
#         0.20 * df["growth_score_norm"] +
#         0.20 * df["hybrid_score_norm"] +
#         0.10 * df["locality_score_norm"] +
#         0.10 * df["rental_yield_norm"]
#     )

#     # -----------------------------
#     # VERDICT
#     # -----------------------------
#     def verdict(row):
#         if row["overall_score"] > 0.7:
#             return "🏆 Best Value"
#         elif row["risk_score"] >= 6:
#             return "⚠️ Risky"
#         elif row["investment_rating"] == "Excellent":
#             return "💰 Strong Investment"
#         elif row["comparison_price_score"] < 0.3:
#             return "💸 Expensive"
#         else:
#             return "👍 Balanced"

#     df["verdict"] = df.apply(verdict, axis=1)

#     def explain(row):
#         reasons = []

#         if row["comparison_price_score"] > 0.6:
#             reasons.append("better price")

#         if row["risk_score"] <= 2:
#             reasons.append("low risk")

#         if row["growth_score"] >= 2:
#             reasons.append("good future growth")

#         return ", ".join(reasons)

#     df["comparison_reason"] = df.apply(explain, axis=1)

#     return df[[
#         "id",
#         "project_name",
#         "location",

#         "price",

#         "risk_score",
#         "growth_score",

#         "hybrid_score",
#         "locality_rating",
#         "rental_yield_percent",
#         "investment_rating",

#         "overall_score",
#         "verdict",
#         "comparison_reason"
#     ]]


#======================================================================


# ===============================
# comparison_agent.py
# ===============================

import pandas as pd


def normalize(series):
    if series.max() == series.min():
        return pd.Series([0.5] * len(series), index=series.index)

    return (series - series.min()) / (
        series.max() - series.min()
    )


def analysis_to_score(flag, severity):
    """
    Convert analysis result into numeric score.

    Higher = better investment value
    """

    flag = str(flag).lower()
    severity = str(severity).lower()

    if flag == "undervalued":
        return 1.0

    if flag == "fair":
        return 0.75

    if flag == "overpriced":

        if severity == "high":
            return 0.0

        return 0.25

    return 0.5


def run_comparison_agent(df):
    """
    Compare properties using:

    - hybrid_score
    - risk_score
    - growth_score
    - rental_yield
    - negotiation_score
    - analysis_score
    - locality_rating

    Returns comparison dataframe.
    """

    print("✅ comparison_agent executed")

    df = df.copy()

    # -----------------------------
    # RISK SCORE
    # -----------------------------
    if "risk_score" in df.columns:
        df["risk_score_norm"] = (
            1 - normalize(df["risk_score"])
        )
    else:
        df["risk_score_norm"] = 0.5

    # -----------------------------
    # GROWTH SCORE
    # -----------------------------
    if "growth_score" in df.columns:
        df["growth_score_norm"] = normalize(
            df["growth_score"]
        )
    else:
        df["growth_score_norm"] = 0.5

    # -----------------------------
    # HYBRID SCORE
    # -----------------------------
    if "hybrid_score" in df.columns:
        df["hybrid_score_norm"] = normalize(
            df["hybrid_score"]
        )
    else:
        df["hybrid_score_norm"] = 0.5

    # -----------------------------
    # LOCALITY SCORE
    # -----------------------------
    if "locality_rating" in df.columns:
        df["locality_score_norm"] = normalize(
            df["locality_rating"]
        )
    else:
        df["locality_score_norm"] = 0.5

    # -----------------------------
    # RENTAL YIELD SCORE
    # -----------------------------
    if "rental_yield_percent" in df.columns:

        rental_clean = (
            df["rental_yield_percent"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .astype(float)
        )

        df["rental_yield_norm"] = normalize(
            rental_clean
        )

    else:
        df["rental_yield_norm"] = 0.5

    # -----------------------------
    # NEGOTIATION SCORE
    # -----------------------------
    if "negotiation_score" in df.columns:

        df["negotiation_score_norm"] = normalize(
            df["negotiation_score"]
        )

    else:
        df["negotiation_score_norm"] = 0.5

    # -----------------------------
    # ANALYSIS SCORE
    # -----------------------------
    if (
        "analysis_flag" in df.columns
        and "analysis_severity" in df.columns
    ):

        df["analysis_score_norm"] = df.apply(
            lambda row: analysis_to_score(
                row["analysis_flag"],
                row["analysis_severity"]
            ),
            axis=1
        )

    else:
        df["analysis_score_norm"] = 0.5

    # -----------------------------
    # FINAL SCORE
    # -----------------------------
    df["overall_score"] = (
        0.25 * df["hybrid_score_norm"] +
        0.20 * df["risk_score_norm"] +
        0.15 * df["growth_score_norm"] +
        0.10 * df["rental_yield_norm"] +
        0.10 * df["negotiation_score_norm"] +
        0.10 * df["analysis_score_norm"] +
        0.10 * df["locality_score_norm"]
    )

    # -----------------------------
    # VERDICT
    # -----------------------------
    def verdict(row):

        if row.get("risk_score", 0) >= 6:
            return "⚠️ Risky"

        if row["overall_score"] >= 0.75:
            return "🏆 Best Value"

        if row.get("analysis_flag") == "undervalued":
            return "💎 Undervalued"

        if row.get("growth_score", 0) >= 3:
            return "🚀 High Growth"

        return "👍 Balanced"

    df["verdict"] = df.apply(
        verdict,
        axis=1
    )

    # -----------------------------
    # EXPLANATION
    # -----------------------------
    def explain(row):

        reasons = []

        if row.get("analysis_flag") == "undervalued":
            reasons.append("undervalued")

        if row.get("risk_score", 0) <= 2:
            reasons.append("low risk")

        if row.get("growth_score", 0) >= 2:
            reasons.append("future growth")

        if row.get("negotiation_score", 0) >= 4:
            reasons.append("high negotiation scope")

        rating = str(
            row.get("investment_rating", "")
        ).lower()

        if rating in ["excellent", "good"]:
            reasons.append("strong rental potential")

        return ", ".join(reasons)

    df["comparison_reason"] = df.apply(
        explain,
        axis=1
    )

    return df[
        [
            "id",
            "project_name",
            "location",
            "price",

            "hybrid_score",
            "risk_score",
            "growth_score",

            "negotiation_score",

            "analysis_flag",
            "analysis_severity",
            "analysis_msg",

            "locality_rating",
            "rental_yield_percent",
            "investment_rating",

            "overall_score",
            "verdict",
            "comparison_reason"
        ]
    ].sort_values(
        "overall_score",
        ascending=False
    )