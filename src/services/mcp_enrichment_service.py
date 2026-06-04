# #==================================
# # mcp_enrichment_service.py
# #==================================

# import pandas as pd

# from src.agents.risk_agent import run_risk_agent
# from src.agents.future_agent import run_future_agent
# from src.agents.rental_agent import run_rental_agent
# from src.recommender.hybrid_recommender import apply_hybrid_ranking
# from src.agents.analysis_agent import run_analysis
# from src.agents.negotiation_agent import run_negotiation_agent


# def enrich_properties(
#     selected_df: pd.DataFrame
# ):

#     df = selected_df.copy()

#     # MCP properties do not come from
#     # cosine similarity search.
#     # Create a default value.
#     df["cosine_similarity"] = 1.0

#     df = apply_hybrid_ranking(
#         df,
#         intent={},
#         slider_weights=None
#     )

#     # -------------------
#     # Analysis
#     # -------------------
#     analysis_results = run_analysis(df)

#     if analysis_results and len(analysis_results) > 0:

#         analysis_df = pd.DataFrame(
#             analysis_results
#         )

#         df = df.merge(
#             analysis_df,
#             on="id",
#             how="left"
#         )


#     # -------------------
#     # Negotiation
#     # -------------------    
#     negotiation_df = run_negotiation_agent(df)

#     if (
#         negotiation_df is not None
#         and len(negotiation_df) > 0
#     ):
#         df = df.merge(
#             negotiation_df,
#             on="id",
#             how="left"
#         )



#     # -------------------
#     # Risk
#     # -------------------
#     risk_results = run_risk_agent(df)

#     if risk_results:
#         risk_df = pd.DataFrame(
#             risk_results
#         )

#         df = df.merge(
#             risk_df,
#             on="id",
#             how="left"
#         )



#     # -------------------
#     # Future
#     # -------------------
#     future_results = run_future_agent(df)

#     if future_results:
#         future_df = pd.DataFrame(
#             future_results
#         )

#         df = df.merge(
#             future_df,
#             on="id",
#             how="left"
#         )



#     # -------------------
#     # Rental
#     # -------------------
#     rental_df = run_rental_agent(df)

#     if (
#         rental_df is not None
#         and len(rental_df) > 0
#     ):

#         df = df.merge(
#             rental_df,
#             on="id",
#             how="left"
#         )

#     return df

#==========================================================================================

#==================================
# mcp_enrichment_service.py
#==================================

import pandas as pd
from src.agents.risk_agent import run_risk_agent
from src.agents.future_agent import run_future_agent
from src.agents.rental_agent import run_rental_agent
from src.agents.analysis_agent import run_analysis
from src.agents.negotiation_agent import run_negotiation_agent
from src.recommender.hybrid_recommender import apply_hybrid_ranking


def enrich_properties(selected_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich selected properties by running them through various analysis agents."""
    
    df = selected_df.copy()
    
    # MCP properties do not come from cosine similarity search; set default.
    df["cosine_similarity"] = 1.0
    df = apply_hybrid_ranking(df, intent={}, slider_weights=None)

    # Define agents to run sequentially
    # Format: (agent_function, name_for_logging/debugging)
    agents = [
        (run_analysis, "analysis"),
        (run_negotiation_agent, "negotiation"),
        (run_risk_agent, "risk"),
        (run_future_agent, "future"),
        (run_rental_agent, "rental")
    ]

    for agent_func, name in agents:
        res = agent_func(df)
        
        # Skip if result is None or an empty collection/dataframe
        if res is None or (isinstance(res, (list, pd.DataFrame)) and len(res) == 0):
            continue
            
        # Convert to DataFrame if the agent returned a list of dicts
        res_df = res if isinstance(res, pd.DataFrame) else pd.DataFrame(res)
        
        # Merge the enrichment data
        df = df.merge(res_df, on="id", how="left")

    return df