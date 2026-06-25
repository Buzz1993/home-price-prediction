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
    print("☑️ prepare_comparison_data executed")

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
    print("☑️ run_comparison executed")

    raw_df = prepare_comparison_data(selected_df) # remove comapare and delete columns and add dev_summary column to selected_df to get raw_df 
    # print("raw_df", raw_df) 
    # print("===============================")


    compare_df = run_comparison_agent(raw_df) #created comparison_price_score, risk_score_norm, growth_score_norm, locality_score_norm, rental_yiend_norm, overall_score, 
                                              #verdict, comparison_reason columns and this function returns compare_df with only selected columns 
                                              # which get shown in "comaparison_result" table in UI 
                                              # that selected columns are "id", "project_name", "location", "price", "risk_score", "growth_score", "hybrid_score"
                                              #"locality_rating", "rental_yield_percent", "investment_rating", "overall_score", "verdict", "comparison_reason".
    # print("compare_df", compare_df) 
    # print("===============================")    

    return raw_df, compare_df


# =============================
# PREPARE MAP DATA
# =============================
def prepare_map_data(df, master_df):
    """
    Make sure every property has latitude and longitude before plotting it on the map.
    """
    print("☑️ prepare_map_data executed")
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