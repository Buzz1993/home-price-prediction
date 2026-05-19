#search_agent.py

from src.data.content_based_filtering import recommend_with_constraints
from src.recommender.hybrid_recommender import apply_hybrid_ranking
from src.agents.analysis_agent import run_analysis
from src.agents.risk_agent import run_risk_agent
from src.agents.future_agent import run_future_agent
from src.agents.negotiation_agent import run_negotiation_agent
from src.agents.rental_agent import run_rental_agent
from src.utils.rent_utils import calculate_rent

import pandas as pd

def add_rent_columns(df):
        df[["min_rent", "max_rent"]] = df.apply(
            lambda row: pd.Series(calculate_rent(row)),
            axis=1
        )
        return df

def run_search_pipeline(df, X_processed, filters, intent, slider_weights, mode):

    recs = recommend_with_constraints(df, X_processed, filters, mode)

    if not recs:
        return None

    # Ranking
    recs["similar"] = apply_hybrid_ranking(
        recs["similar"], intent, slider_weights
    )

    recs["similar"] = add_rent_columns(recs["similar"])

    # Agents
    analysis_results = run_analysis(recs["similar"])
    risk_results = run_risk_agent(recs["similar"])
    future_results = run_future_agent(recs["similar"])


    # Merge future
    if future_results and len(future_results) > 0:
        future_df = pd.DataFrame(future_results)
        recs["similar"] = recs["similar"].merge(future_df, on="id", how="left")

    # Merge risk
    if risk_results:
        risk_df = pd.DataFrame(risk_results)
        recs["similar"] = recs["similar"].merge(risk_df, on="id", how="left")

    # Safety
    for col in ["growth_label", "growth_reason"]:
        if col not in recs["similar"].columns:
            recs["similar"][col] = None

    # Analysis mapping
    analysis_map = {a["id"]: a for a in analysis_results}

    recs["similar"]["analysis_flag"] = recs["similar"]["id"].map(
        lambda x: analysis_map.get(x, {}).get("analysis_flag")
    )

    recs["similar"]["analysis_msg"] = recs["similar"]["id"].map(
        lambda x: analysis_map.get(x, {}).get("analysis_msg")
    )

    recs["similar"]["analysis_severity"] = recs["similar"]["id"].map(
        lambda x: analysis_map.get(x, {}).get("analysis_severity")
    )

    # -----------------------------
    # NEGOTIATION AGENT (ADD HERE)
    # -----------------------------
    negotiation_df = run_negotiation_agent(recs["similar"])

    if negotiation_df is not None and len(negotiation_df) > 0:
        recs["similar"] = recs["similar"].merge(
            negotiation_df,
            on="id",
            how="left"
        )

    # -----------------------------
    # RENTAL AGENT (NEW)
    # -----------------------------
    rental_df = run_rental_agent(recs["similar"])

    if rental_df is not None and len(rental_df) > 0:
        recs["similar"] = recs["similar"].merge(
            rental_df,
            on="id",
            how="left"
        )

    return recs