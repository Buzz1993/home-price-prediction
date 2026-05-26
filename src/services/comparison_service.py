#src/services/comparison_service.py
# ===============================
# comparison_service.py
# ===============================

import pandas as pd

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
    - removing UI columns : "Compare" and "Delete"
    - adding development summary

    Returns cleaned and enriched dataframe.
    """

    # selected_df is the dataframe of selected properties
    # with only rows where Compare column is True
    df = selected_df.copy()

    # Remove UI column
    if "Compare" in df.columns:
        df = df.drop(columns=["Compare"])

    # Remove UI delete column
    if "Delete" in df.columns:
        df = df.drop(columns=["Delete"])

    # DEVELOPMENT SUMMARY
    df["dev_summary"] = df.apply(
        lambda row: get_development_summary(
            row["location"],
            row["city"]
        ),
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

    raw_df = prepare_comparison_data(selected_df) # raw_df is the enriched dataframe of selected properties with calculated rent, risk_score, growth_score, and development summary. This is the input to the comparison agent. 
    # print("raw_df", raw_df) 
    # print("===============================")


    compare_df = run_comparison_agent(raw_df) # from raw_df, we get compare_df which is the final comparison result dataframe with columns like "id", "project_name", "location", "price", "risk_score", "growth_score", "hybrid_score", "locality_rating", "rental_yield_percent", "investment_rating", "overall_score", "verdict", and "explanation" (short explanation of why a property got a certain verdict based on its scores)
                                                    #only overall_score column is created in run_comparison_agent, else note that other columns are already in prepared_df, we only take some of them in comapre_df for showing them in comaparison Result table in UI.
    # print("compare_df", compare_df) 
    # print("===============================")    

    return raw_df, compare_df


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