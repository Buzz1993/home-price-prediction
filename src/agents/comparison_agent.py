# ===============================
# comparison_agent.py
# ===============================
import numpy as np
import pandas as pd

def normalize(series):
    if series.max() == series.min():
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def run_comparison_agent(df):
    """
    Compare selected properties using:
    - price_score
    - risk_score
    - growth_score

    Calculates:
    - overall_score
    - verdict
    - short explanation

    All these values are stored inside
    the comparison result dataframe.

    Returns final comparison dataframe.
    """
    print("✅ comparison_agent executed")
    df = df.copy()

    # -----------------------------
    # PRICE SCORE (lower is better)
    # -----------------------------
    df["price_score"] = 1 - normalize(df["price"])

    # -----------------------------
    # RISK SCORE (lower is better)
    # -----------------------------
    if "risk_score" in df.columns:
        df["risk_score_norm"] = 1 - normalize(df["risk_score"])
    else:
        df["risk_score_norm"] = 0.5

    # -----------------------------
    # GROWTH SCORE (higher is better)
    # -----------------------------
    if "growth_score" in df.columns:
        df["growth_score_norm"] = normalize(df["growth_score"])
    else:
        df["growth_score_norm"] = 0.5

    # -----------------------------
    # FINAL SCORE
    # -----------------------------
    df["overall_score"] = (
        0.4 * df["price_score"] +
        0.3 * df["risk_score_norm"] +
        0.3 * df["growth_score_norm"]
    )

    # -----------------------------
    # VERDICT
    # -----------------------------
    def verdict(row):
        if row["overall_score"] > 0.7:
            return "🏆 Best Value"
        elif row["risk_score"] >= 6:
            return "⚠️ Risky"
        elif row["price_score"] < 0.3:
            return "💸 Expensive"
        else:
            return "👍 Balanced"

    df["verdict"] = df.apply(verdict, axis=1)

    def explain(row):
        reasons = []

        if row["price_score"] > 0.6:
            reasons.append("better price")

        if row["risk_score"] <= 2:
            reasons.append("low risk")

        if row["growth_score"] >= 2:
            reasons.append("good future growth")

        return ", ".join(reasons)

    df["explanation"] = df.apply(explain, axis=1)

    return df[[
        "id",
        "price",
        "risk_score",
        "growth_score",
        "overall_score",
        "verdict",
        "explanation"   
    ]]