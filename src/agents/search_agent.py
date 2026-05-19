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

    recs = recommend_with_constraints(df, X_processed, filters, mode) #get "input" and "similar" properties based on filters and mode

    if not recs:
        return None

    # Ranking
    recs["similar"] = apply_hybrid_ranking(recs["similar"], intent, slider_weights) #rank similar properties based on hybrid score (cosine similarity + weighted business score) and 
                                                                                    #add column as "hybrid_score" to Similar Properties dataframe

    recs["similar"] = add_rent_columns(recs["similar"]) #add min_rent and max_rent columns to Similar Properties dataframe by applying calculate_rent function on each row

    # Agents
    analysis_results = run_analysis(recs["similar"]) #run analysis agent on similar properties to get analysis results with columns "id", "analysis_flag", "analysis_msg", "analysis_severity"
    risk_results = run_risk_agent(recs["similar"]) #run risk agent on similar properties to get risk results with columns "id", "risk_categories", "risk_score", "risk_label"
    future_results = run_future_agent(recs["similar"]) #run future agent on similar properties to get future results with columns "id", "growth_label", "growth_reason"


    # Merge future
    #check if future_results exists and is not empty, then create future_df from future_results and merge with recs["similar"] on "id" column to add future agent results to Similar Properties dataframe
    if future_results and len(future_results) > 0: 
        future_df = pd.DataFrame(future_results)
        recs["similar"] = recs["similar"].merge(future_df, on="id", how="left") #future_df has columns "id", "growth_label", "growth_reason", "growth_score" which will be added to Similar Properties dataframe based on matching "id" values 

    # Merge risk
    if risk_results:
        risk_df = pd.DataFrame(risk_results)
        recs["similar"] = recs["similar"].merge(risk_df, on="id", how="left") #risk_df has columns "id", "risk_categories", "risk_score", "risk_label" which will be added to Similar Properties dataframe based on matching "id" values

    # Safety
    for col in ["growth_label", "growth_reason"]: #check if growth_label and growth_reason columns exist in recs["similar"] dataframe, if not, then create them with None values to ensure these columns are always present for downstream processing
        if col not in recs["similar"].columns:
            recs["similar"][col] = None 

    # Analysis mapping
    analysis_map = {a["id"]: a for a in analysis_results} #create a dictionary analysis_map where keys are property "id" from analysis_results and values are the corresponding analysis result dictionaries, this allows for easy lookup of analysis results by property id
    
    #Merge analysis results into Similar Properties dataframe by mapping "id" column to analysis_map to get corresponding 
    #"analysis_flag", "analysis_msg", "analysis_severity" values for each property, if id does not exist in analysis_map then set these columns to None

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
    # Run negotiation agent on similar properties to get negotiation results with columns "id", "negotiation_tips", "negotiation_score"
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

    return recs