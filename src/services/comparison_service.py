#src/services/comparison_service.py
# ===============================
# comparison_service.py
# ===============================

import pandas as pd

from src.utils.rent_utils import calculate_rent
from src.utils.development_utils import get_development_summary
from src.agents.comparison_agent import run_comparison_agent


# =============================
# PREPARE DATA FOR COMPARISON
# =============================
def prepare_comparison_data(selected_df):
    # =============================
    # Clean and enrich selected properties
    # =============================
    """
    Prepare selected properties for comparison by:
    - removing UI columns
    - calculating rent
    - creating risk_score from risk_label
    - creating growth_score from growth_label
    - adding development summary

    Returns cleaned and enriched dataframe.
    """

    df = selected_df.copy()

    # Remove UI column
    if "Compare" in df.columns:
        df = df.drop(columns=["Compare"])

    # RENT CALCULATION
    def safe_rent(row):
        try:
            min_rent, max_rent = calculate_rent(row)
            return pd.Series([min_rent or 0, max_rent or 0])
        except:
            return pd.Series([0, 0])

    df[["min_rent", "max_rent"]] = df.apply(safe_rent, axis=1)

    # RISK SCORE
    if "risk_score" not in df.columns:
        df["risk_score"] = df["risk_label"].map({
            "🟢 Low Risk": 1,
            "🟡 Medium Risk": 4,
            "🔴 High Risk": 7
        }).fillna(3)

    # GROWTH SCORE
    if "growth_score" not in df.columns:
        df["growth_score"] = df["growth_label"].map({
            "🚀 High Growth": 3,
            "📍 Mature Area": 1,
            "➖ No Growth Signal": 0
        }).fillna(1)

    # DEVELOPMENT SUMMARY
    df["dev_summary"] = df.apply(
        lambda row: get_development_summary(row["location"], row["city"]),
        axis=1
    )

    return df


# =============================
# RUN COMPARISON ENGINE
# =============================
def run_comparison(selected_df):
    # =============================
    # Execute comparison scoring
    # =============================
    """
    Runs comparison agent on prepared data.
    """

    prepared_df = prepare_comparison_data(selected_df) # prepared_df is the enriched dataframe of selected properties with calculated rent, risk_score, growth_score, and development summary. This is the input to the comparison agent. 
    # print("prepared_df", prepared_df) 
    # print("===============================")


    compare_df = run_comparison_agent(prepared_df) # Comparison Result dataframe : includes the original selected property data along with new columns like price_score, risk_score_norm, growth_score_norm, overall_score, verdict, and explanation_msg
    # print("compare_df", compare_df) 
    # print("===============================")    

    return prepared_df, compare_df


# =============================
# PREPARE MAP DATA
# =============================
def prepare_map_data(df, master_df):
    """
    Ensure latitude & longitude exist.
    If already present → use directly
    Else → fallback to merge
    """

    # ✅ CASE 1: Already present (BEST CASE)
    if "latitude" in df.columns and "longitude" in df.columns:
        return df

    # ✅ CASE 2: Try merging (fallback)
    master_df = master_df.rename(columns={
        "lattitude": "latitude",
        "longtitude": "longitude"
    })

    merged = df.merge(
        master_df[["id", "latitude", "longitude"]],
        on="id",
        how="left"
    )

    return merged