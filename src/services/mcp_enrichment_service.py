#==================================
# mcp_enrichment_service.py
#==================================

import pandas as pd

from src.agents.risk_agent import run_risk_agent
from src.agents.future_agent import run_future_agent
from src.agents.rental_agent import run_rental_agent
from src.recommender.hybrid_recommender import apply_hybrid_ranking
from src.agents.analysis_agent import run_analysis
from src.agents.negotiation_agent import run_negotiation_agent


def enrich_properties(
    selected_df: pd.DataFrame
):

    df = selected_df.copy()

    # MCP properties do not come from
    # cosine similarity search.
    # Create a default value.
    df["cosine_similarity"] = 1.0

    df = apply_hybrid_ranking(
        df,
        intent={},
        slider_weights=None
    )

    # -------------------
    # Analysis
    # -------------------
    analysis_results = run_analysis(df)

    if analysis_results and len(analysis_results) > 0:

        analysis_df = pd.DataFrame(
            analysis_results
        )

        df = df.merge(
            analysis_df,
            on="id",
            how="left"
        )


    # -------------------
    # Negotiation
    # -------------------    
    negotiation_df = run_negotiation_agent(df)

    if (
        negotiation_df is not None
        and len(negotiation_df) > 0
    ):
        df = df.merge(
            negotiation_df,
            on="id",
            how="left"
        )



    # -------------------
    # Risk
    # -------------------
    risk_results = run_risk_agent(df)

    if risk_results:
        risk_df = pd.DataFrame(
            risk_results
        )

        df = df.merge(
            risk_df,
            on="id",
            how="left"
        )



    # -------------------
    # Future
    # -------------------
    future_results = run_future_agent(df)

    if future_results:
        future_df = pd.DataFrame(
            future_results
        )

        df = df.merge(
            future_df,
            on="id",
            how="left"
        )



    # -------------------
    # Rental
    # -------------------
    rental_df = run_rental_agent(df)

    if (
        rental_df is not None
        and len(rental_df) > 0
    ):

        df = df.merge(
            rental_df,
            on="id",
            how="left"
        )

    return df