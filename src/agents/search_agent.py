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
    if df.empty:
        return df

    #add estimated_rent_min and estimated_rent_max columns to Similar Properties dataframe by applying calculate_rent function on each row
    df[["estimated_rent_min", "estimated_rent_max"]] = df.apply(
        lambda row: pd.Series(calculate_rent(row)),
        axis=1
    )
    return df


def run_search_pipeline(df, X_processed, filters, intent, slider_weights, mode):
    """
    Run complete property recommendation pipeline
    including filtering, ranking, and all agents.

    Returns:
        dict -> {"input", "similar"}
    """
    
    print("✅ search_agent executed")

    recs = recommend_with_constraints(df, X_processed, filters, mode) #get "input" and "similar" properties based on filters and mode

    if not recs:
        return None

    # Ranking
    recs["similar"] = apply_hybrid_ranking(similar_df=recs["similar"], intent=intent, slider_weights=slider_weights) #rank similar properties based on hybrid score (cosine similarity + weighted business score) and 
                                                                                    #add column as "hybrid_score" to Similar Properties dataframe

    print("1"*50)
    print("SIMILAR DATAFRAME COLUMNS",recs["similar"].columns.tolist())
    print("1"*50)

    recs["similar"] = add_rent_columns(recs["similar"]) #add estimated_rent_min and estimated_rent_max columns to Similar Properties dataframe by applying calculate_rent function on each row

    print("2"*50)
    print("SIMILAR DATAFRAME COLUMNS",recs["similar"].columns.tolist())
    print("2"*50)

    # Agents
    analysis_results = run_analysis(recs["similar"]) #run analysis agent on similar properties to get analysis results in list of dict - "id", "analysis_flag", "analysis_msg", "analysis_severity"
    risk_results = run_risk_agent(recs["similar"]) #run risk agent on similar properties to get risk results in list of dict - "id", "risk_categories", "risk_score", "risk_label"
    future_results = run_future_agent(recs["similar"]) #run future agent on similar properties to get future results in list of dict - "id", "growth_label", "growth_reason", "future_signals", "infra_detected"


    # Merge future
    #check if future_results exists and is not empty, then create future_df from future_results and merge with recs["similar"] on "id" column to add future agent results to Similar Properties dataframe
    if future_results and len(future_results) > 0: 
        future_df = pd.DataFrame(future_results)  # Convert list of dictionaries into pandas DataFrame.  
        recs["similar"] = recs["similar"].merge(future_df, on="id", how="left") #future_df has columns "id", "growth_label", "growth_reason", "growth_score", "future_signals", "infra_detected"  which will be added to Similar Properties dataframe based on matching "id" values 

    # Merge risk
    if risk_results:
        risk_df = pd.DataFrame(risk_results)
        recs["similar"] = recs["similar"].merge(risk_df, on="id", how="left") #risk_df has columns "id", "risk_categories", "risk_score", "risk_label" which will be added to Similar Properties dataframe based on matching "id" values


    # Merge analysis results
    if analysis_results and len(analysis_results) > 0:
        analysis_df = pd.DataFrame(analysis_results) # Convert list of dictionaries into DataFrame
        recs["similar"] = recs["similar"].merge(analysis_df,on="id",how="left") # Merge analysis results into Similar Properties dataframe

    # -----------------------------
    # NEGOTIATION AGENT (ADD HERE)
    # -----------------------------
    # Run negotiation agent on similar properties to get negotiation results with columns "id","negotiation_power",-
    # "suggested_discount_percent","target_price","price_position","strategy","talking_points"
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
    # Run rental agent on similar properties to get rental results with columns "id", "rent_estimate", "rent_reasoning"
    rental_df = run_rental_agent(recs["similar"])

    if rental_df is not None and len(rental_df) > 0:
        recs["similar"] = recs["similar"].merge(
            rental_df,
            on="id",
            how="left"
        )

    return recs #return input and similar properties