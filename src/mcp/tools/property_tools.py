# # =====================================================================
# # src/mcp/tools/property_tools.py
# # =====================================================================

# import json
# from src.services.mcp_real_estate_service import (
#     run_mcp_comparison,
#     run_mcp_rental,
#     run_mcp_prediction,
#     run_mcp_negotiation,
#     run_mcp_valuation,
#     run_mcp_advisor
# )


# # =====================================================================
# # 1. INVESTMENT COMPARISON & RANKING
# # =====================================================================
# def compare_properties(property_ids: list[str]) -> str:
#     """Compare multiple properties and return investment ranking scores and verdicts."""
#     if len(property_ids) < 2:
#         return json.dumps({"error": "Need at least 2 properties for analytical comparison"}, indent=2)

#     raw_df, compare_df = run_mcp_comparison(property_ids)

#     if compare_df.empty:
#         return json.dumps({"error": "Comparison returned no results"}, indent=2)

#     # Sort to determine rankings and extract the clear winner
#     compare_df = compare_df.sort_values("overall_score", ascending=False)
    
#     ranking_cols = ["id", "overall_score", "verdict", "comparison_reason"]
#     rankings = compare_df[ranking_cols].to_dict(orient="records")
    
#     result = {
#         "winner": rankings[0],
#         "rankings": rankings
#     }
#     return json.dumps(result, indent=2, default=str)


# # =====================================================================
# # 2. RENTAL MATRIX ANALYTICS
# # =====================================================================
# def get_rental_analysis(property_ids: list[str]) -> str:
#     """Run rental yield analysis, estimates, and demand metrics for given properties."""
#     if not property_ids:
#         return json.dumps({"error": "No properties provided for rental analysis"}, indent=2)

#     rental_df = run_mcp_rental(property_ids)
#     return json.dumps(
#         rental_df.to_dict(orient="records"), 
#         indent=2, 
#         default=str
#     )


# # =====================================================================
# # 3. ML PRICE PREDICTION MODEL
# # =====================================================================
# def get_price_prediction(property_ids: list[str]) -> str:
#     """Invokes prediction engine models to forecast valuation pricing differences."""
#     if not property_ids:
#         return json.dumps({"error": "No properties provided for price prediction"}, indent=2)
        
#     prediction_df = run_mcp_prediction(property_ids)
#     return json.dumps(
#         prediction_df.to_dict(orient="records"),
#         indent=2,
#         default=str
#     )


# # =====================================================================
# # 4. NEGOTIATION STRATEGY GUIDE
# # =====================================================================
# def get_negotiation_strategy(property_ids: list[str]) -> str:
#     """Generates localized buyer leverage power, target prices, and strategic talking points."""
#     if not property_ids:
#         return json.dumps({"error": "No properties available for strategy mapping"}, indent=2)
        
#     negotiation_df = run_mcp_negotiation(property_ids)
#     return json.dumps(
#         negotiation_df.to_dict(orient="records"),
#         indent=2,
#         default=str
#     )


# # =====================================================================
# # 5. BENCHMARK VALUATION ANALYTICS
# # =====================================================================
# def get_valuation_analysis(property_ids: list[str]) -> str:
#     """Evaluates core market benchmarking thresholds to flag fair-market pricing deviations."""
#     if not property_ids:
#         return json.dumps({"error": "No target records isolated for pricing evaluation"}, indent=2)
        
#     valuation_df = run_mcp_valuation(property_ids)
#     return json.dumps(
#         valuation_df.to_dict(orient="records"),
#         indent=2,
#         default=str
#     )


# # =====================================================================
# # 6. PORTFOLIO INVESTMENT ADVISOR
# # =====================================================================
# def get_investment_advice(property_ids: list[str]) -> str:
#     """Runs high-conviction decision scoring matrices to flag risks, positives, and buyer profiles."""
#     if not property_ids:
#         return json.dumps({"error": "No properties staged for investment advising"}, indent=2)
        
#     advisor_df = run_mcp_advisor(property_ids)
#     return json.dumps(
#         advisor_df.to_dict(orient="records"),
#         indent=2,
#         default=str
#     )


#==========================================================================================================================================================================

# """
# Unified Real Estate Analytical Tools Layer.
# Enforces whitelists across search results and batch detail projections to maintain token efficiency.
# """
# import pandas as pd
# from src.core.search_registry import SEARCH_STATE, GLOBAL_MASTER_DF, CACHED_SEARCH_METADATA
# from src.utils.search_engine import query
# from src.services.mcp_real_estate_service import (
#     run_mcp_comparison,
#     run_mcp_rental,
#     run_mcp_prediction,
#     run_mcp_negotiation,
#     run_mcp_valuation,
#     run_mcp_advisor,
#     clear_enrichment_cache
# )

# # Tight field filters to optimize context window storage
# SEARCH_RESULTS_WHITELIST = [
#     "id", "project_name", "price", "location", "bhk_type", "amenities_mcp", "search_score"
# ]

# PROPERTY_DETAIL_WHITELIST = [
#     "id", "project_name", "builder", "location", "price", "area", 
#     "bhk_type", "amenities_mcp", "features_mcp", "analysis_msg"
# ]

# # =====================================================================
# # 1. DISCOVERY & BULK DATA EXTRACTION RETRIEVAL TOOLS
# # =====================================================================

# def search_properties(bhk: str = None, amenities: str = None, location: str = None, limit: int = 5) -> list[dict]:
#     """Query inventory through fast text tokens and exact matching structural masks."""
#     extracted_criteria = {
#         "bhk": f"{bhk.strip().lower().replace(' ', '')}" if bhk else None,
#         "amenities": amenities,
#         "location": location
#     }
    
#     active_criteria = {k: v for k, v in extracted_criteria.items() if v}
#     if not active_criteria:
#         return []
        
#     determined_min_matches = 1 if (not amenities or not location) else 2
#     results_df = query(SEARCH_STATE, extracted_criteria, min_matches=determined_min_matches)
    
#     if results_df.empty:
#         return []
        
#     if extracted_criteria["bhk"] and "bhk_type" in results_df.columns:
#         target_bhk = extracted_criteria["bhk"]
#         filtered_df = results_df[
#             results_df["bhk_type"].str.lower().str.replace(" ", "") == target_bhk
#         ]
#         if not filtered_df.empty:
#             results_df = filtered_df

#     out_cols = [c for c in SEARCH_RESULTS_WHITELIST if c in results_df.columns]
#     return results_df[out_cols].head(limit).to_dict(orient="records")


# def get_property_details(property_id: str) -> dict:
#     """Perform a filtered drill-down isolation query matching a single asset ID."""
#     target_id = str(property_id).strip()
#     match_frame = GLOBAL_MASTER_DF[GLOBAL_MASTER_DF["id"].astype(str).str.strip() == target_id]
    
#     if match_frame.empty:
#         return {"error": f"No active recorded data elements matching individual asset key '{target_id}' could be isolated."}
        
#     out_cols = [c for c in PROPERTY_DETAIL_WHITELIST if c in match_frame.columns]
#     return match_frame[out_cols].to_dict(orient="records")[0]


# def get_properties_by_ids(property_ids: list[str]) -> list[dict]:
#     """Batch-loads and resolves multiple property detail records in a single round-trip call."""
#     target_ids = [str(pid).strip() for pid in property_ids if pid]
#     if not target_ids:
#         return []
        
#     match_frame = GLOBAL_MASTER_DF[GLOBAL_MASTER_DF["id"].astype(str).str.strip().isin(target_ids)]
#     if match_frame.empty:
#         return []
        
#     out_cols = [c for c in PROPERTY_DETAIL_WHITELIST if c in match_frame.columns]
#     return match_frame[out_cols].to_dict(orient="records")


# # =====================================================================
# # 2. PROPERTY EVALUATION TOOLS
# # =====================================================================

# def compare_properties(property_ids: list[str]) -> dict:
#     """Compare multiple properties side-by-side using advanced stacking matrix scores."""
#     target_ids = [str(pid).strip() for pid in property_ids if pid]
#     if len(target_ids) < 2:
#         return {"error": "Analytical stacking comparison metrics require at least 2 distinct property IDs."}

#     _, compare_df = run_mcp_comparison(target_ids)
#     if compare_df.empty:
#         return {"error": "Comparison pipeline execution returned an empty validation frame."}

#     compare_df = compare_df.sort_values("overall_score", ascending=False)
#     ranking_cols = ["id", "overall_score", "verdict", "comparison_reason"]
#     rankings = compare_df[ranking_cols].to_dict(orient="records")
    
#     return {
#         "winner": rankings[0],
#         "rankings": rankings
#     }


# def get_rental_analysis(property_ids: list[str]) -> list[dict]:
#     """Extract micro-rental income generation metrics, estimates, and yields."""
#     target_ids = [str(pid).strip() for pid in property_ids if pid]
#     if not target_ids:
#         return []
#     return run_mcp_rental(target_ids).to_dict(orient="records")


# def get_price_prediction(property_ids: list[str]) -> list[dict]:
#     """Involves prediction core server logic routines to trace estimated variance metrics."""
#     target_ids = [str(pid).strip() for pid in property_ids if pid]
#     if not target_ids:
#         return []
#     return run_mcp_prediction(target_ids).to_dict(orient="records")


# def get_negotiation_strategy(property_ids: list[str]) -> list[dict]:
#     """Generates localized leverage power scores, target caps, and verbal counter strategies."""
#     target_ids = [str(pid).strip() for pid in property_ids if pid]
#     if not target_ids:
#         return []
#     return run_mcp_negotiation(target_ids).to_dict(orient="records")


# def get_valuation_analysis(property_ids: list[str]) -> list[dict]:
#     """Validates baseline distribution thresholds to identify fair market pricing parameters."""
#     target_ids = [str(pid).strip() for pid in property_ids if pid]
#     if not target_ids:
#         return []
#     return run_mcp_valuation(target_ids).to_dict(orient="records")


# def get_investment_advice(property_ids: list[str]) -> list[dict]:
#     """Runs structural profile matrices to chart risk metrics against investment horizons."""
#     target_ids = [str(pid).strip() for pid in property_ids if pid]
#     if not target_ids:
#         return []
#     return run_mcp_advisor(target_ids).to_dict(orient="records")


# def clear_property_analysis_cache() -> dict:
#     """Explicitly purges agent model enrichment caches while maintaining the immutable engine states intact."""
#     clear_enrichment_cache()
#     return {"status": "success", "message": "Global downstream property agent enrichment caches successfully flushed."}

#======================================================================================================================================================================
# # ==============================
# # src/mcp/tools/property_tools.py
# # ==============================
# """
# Unified Real Estate Analytical Tools Layer.
# Enforces whitelists across search results and batch detail projections to maintain token efficiency.
# """
# import pandas as pd
# import requests
# from src.core.search_registry import SEARCH_STATE, GLOBAL_MASTER_DF, CACHED_SEARCH_METADATA
# from src.utils.search_engine import query
# from src.services.mcp_real_estate_service import (
#     run_mcp_comparison,
#     run_mcp_rental,
#     run_mcp_prediction,
#     run_mcp_negotiation,
#     run_mcp_valuation,
#     run_mcp_advisor,
#     clear_enrichment_cache
# )

# # Tight field filters to optimize context window storage
# SEARCH_RESULTS_WHITELIST = [
#     "id", "project_name", "price", "location", "bhk_type", "amenities_mcp", "search_score"
# ]

# PROPERTY_DETAIL_WHITELIST = [
#     "id", "project_name", "builder", "location", "price", "area", 
#     "bhk_type", "amenities_mcp", "features_mcp", "analysis_msg"
# ]

# # =====================================================================
# # 1. DISCOVERY & BULK DATA EXTRACTION RETRIEVAL TOOLS
# # =====================================================================

# def search_properties(bhk: str = None, amenities: str = None, location: str = None, limit: int = 5) -> list[dict]:
#     """Query inventory through fast text tokens and exact matching structural masks."""
#     extracted_criteria = {
#         "bhk": f"{bhk.strip().lower().replace(' ', '')}" if bhk else None,
#         "amenities": amenities,
#         "location": location
#     }
    
#     active_criteria = {k: v for k, v in extracted_criteria.items() if v}
#     if not active_criteria:
#         return []
        
#     determined_min_matches = 1 if (not amenities or not location) else 2
#     results_df = query(SEARCH_STATE, extracted_criteria, min_matches=determined_min_matches)

#     print("\n===== QUERY OUTPUT =====")

#     print("Criteria:", extracted_criteria)
#     print("Rows Returned:", len(results_df))

#     if not results_df.empty:
#         cols = [c for c in ["id", "location", "city", "bhk_type"] if c in results_df.columns]
#         print(results_df[cols].head(10))

#     print("========================\n")
    
#     if results_df.empty:
#         return []
    
        
#     if extracted_criteria["bhk"] and "bhk_type" in results_df.columns:
#         target_bhk = extracted_criteria["bhk"]
#         filtered_df = results_df[
#             results_df["bhk_type"].str.lower().str.replace(" ", "") == target_bhk
#         ]
#         if not filtered_df.empty:
#             results_df = filtered_df

#             print("\n===== AFTER BHK FILTER =====")
#             print("Rows:", len(results_df))

#             cols = [c for c in ["id", "location", "city", "bhk_type"] if c in results_df.columns]
#             print(results_df[cols].head(10))

#             print("============================\n")

#     out_cols = [c for c in SEARCH_RESULTS_WHITELIST if c in results_df.columns]
#     return results_df[out_cols].head(limit).to_dict(orient="records")


# def get_property_details(property_id: str) -> dict:
#     """Perform a filtered drill-down isolation query matching a single asset ID."""
#     target_id = str(property_id).strip()
#     match_frame = GLOBAL_MASTER_DF[GLOBAL_MASTER_DF["id"].astype(str).str.strip() == target_id]
    
#     if match_frame.empty:
#         return {"error": f"No active recorded data elements matching individual asset key '{target_id}' could be isolated."}
        
#     out_cols = [c for c in PROPERTY_DETAIL_WHITELIST if c in match_frame.columns]
#     return match_frame[out_cols].to_dict(orient="records")[0]


# def get_properties_by_ids(property_ids: list[str]) -> list[dict]:
#     """Batch-loads and resolves multiple property detail records in a single round-trip call."""
#     target_ids = [str(pid).strip() for pid in property_ids if pid]
#     if not target_ids:
#         return []
        
#     match_frame = GLOBAL_MASTER_DF[GLOBAL_MASTER_DF["id"].astype(str).str.strip().isin(target_ids)]
#     if match_frame.empty:
#         return []
        
#     out_cols = [c for c in PROPERTY_DETAIL_WHITELIST if c in match_frame.columns]
#     return match_frame[out_cols].to_dict(orient="records")


# # =====================================================================
# # 2. PROPERTY EVALUATION TOOLS
# # =====================================================================

# def compare_properties(property_ids: list[str]) -> dict:
#     """
#     Compare multiple properties using MCP analysis.
#     Sort properties by overall_score.
#     Return the highest-scoring property as the winner
#     along with the complete ranking list.
#     """
#     target_ids = [str(pid).strip() for pid in property_ids if pid]
#     if len(target_ids) < 2:
#         return {"error": "Analytical stacking comparison metrics require at least 2 distinct property IDs."}

#     _, compare_df = run_mcp_comparison(target_ids)
#     if compare_df.empty:
#         return {"error": "Comparison pipeline execution returned an empty validation frame."}

#     compare_df = compare_df.sort_values("overall_score", ascending=False)
#     ranking_cols = ["id", "overall_score", "verdict", "comparison_reason"] #Keep only important columns.
#     rankings = compare_df[ranking_cols].to_dict(orient="records") #Convert dataframe → list of dictionaries.
    
#     return {
#         "winner": rankings[0],
#         "rankings": rankings
#     }


# def get_rental_analysis(property_ids: list[str]) -> list[dict]:
#     """Extract micro-rental income generation metrics, estimates, and yields."""
#     target_ids = [str(pid).strip() for pid in property_ids if pid]
#     if not target_ids:
#         return []
#     return run_mcp_rental(target_ids).to_dict(orient="records")


# def get_price_prediction(property_ids: list[str]) -> list[dict]:
#     """Involves prediction core server logic routines to trace estimated variance metrics."""
#     target_ids = [str(pid).strip() for pid in property_ids if pid]
#     if not target_ids:
#         return []
#     return run_mcp_prediction(target_ids).to_dict(orient="records")


# def get_negotiation_strategy(property_ids: list[str]) -> list[dict]:
#     """Generates localized leverage power scores, target caps, and verbal counter strategies."""
#     target_ids = [str(pid).strip() for pid in property_ids if pid]
#     if not target_ids:
#         return []
#     return run_mcp_negotiation(target_ids).to_dict(orient="records")


# def get_valuation_analysis(property_ids: list[str]) -> list[dict]:
#     """Validates baseline distribution thresholds to identify fair market pricing parameters."""
#     target_ids = [str(pid).strip() for pid in property_ids if pid]
#     if not target_ids:
#         return []
#     return run_mcp_valuation(target_ids).to_dict(orient="records")


# def get_investment_advice(property_ids: list[str]) -> list[dict]:
#     """Runs structural profile matrices to chart risk metrics against investment horizons."""
#     target_ids = [str(pid).strip() for pid in property_ids if pid]
#     if not target_ids:
#         return []
#     return run_mcp_advisor(target_ids).to_dict(orient="records")


# def clear_property_analysis_cache() -> dict:
#     """Explicitly purges agent model enrichment caches while maintaining the immutable engine states intact."""
#     clear_enrichment_cache()
#     return {"status": "success", "message": "Global downstream property agent enrichment caches successfully flushed."}

# # =====================================================================
# # 3. N8N REPORT DELIVERY
# # =====================================================================

# def send_property_report(phone_number: str, report: str) -> dict:
#     """
#     Sends property report to n8n workflow.
#     """

#     webhook_url = "https://buzz123.app.n8n.cloud/webhook-test/98d3e0b7-9577-43ff-8f7a-a43956739ff9"

#     response = requests.post(
#         webhook_url,
#         json={
#             "phone": phone_number,
#             "report": report
#         },
#         timeout=30
#     )

#     return {
#         "status": "success",
#         "status_code": response.status_code
#     }

#======================================================================================================================================================================

# ==============================
# src/mcp/tools/property_tools.py
# ==============================
"""
Unified Real Estate Analytical Tools Layer.
Enforces whitelists across search results and batch detail projections to maintain token efficiency.
"""
import pandas as pd
import requests
from src.core.search_registry import SEARCH_STATE, GLOBAL_MASTER_DF, CACHED_SEARCH_METADATA
from src.utils.search_engine import query
from src.services.mcp_real_estate_service import (
    run_mcp_comparison,
    run_mcp_rental,
    run_mcp_prediction,
    run_mcp_negotiation,
    run_mcp_valuation,
    run_mcp_advisor,
    clear_enrichment_cache
)

# Tight field filters to optimize context window storage
SEARCH_RESULTS_WHITELIST = [
    "id", "project_name", "price", "location", "bhk_type", "amenities_mcp", "search_score"
]

PROPERTY_DETAIL_WHITELIST = [
    "id", "project_name", "builder", "location", "price", "area", 
    "bhk_type", "amenities_mcp", "features_mcp", "analysis_msg"
]

# =====================================================================
# 1. DISCOVERY & BULK DATA EXTRACTION RETRIEVAL TOOLS
# =====================================================================

def search_properties(bhk: str = None, amenities: str = None, location: str = None, limit: int = 5) -> list[dict]:
    """Query inventory through fast text tokens and exact matching structural masks."""
    extracted_criteria = {
        "bhk": f"{bhk.strip().lower().replace(' ', '')}" if bhk else None,
        "amenities": amenities,
        "location": location
    }
    
    active_criteria = {k: v for k, v in extracted_criteria.items() if v}
    if not active_criteria:
        return []
        
    determined_min_matches = 1 if (not amenities or not location) else 2
    results_df = query(SEARCH_STATE, extracted_criteria, min_matches=determined_min_matches)

    print("\n===== QUERY OUTPUT =====")

    print("Criteria:", extracted_criteria)
    print("Rows Returned:", len(results_df))

    if not results_df.empty:
        cols = [c for c in ["id", "location", "city", "bhk_type"] if c in results_df.columns]
        print(results_df[cols].head(10))

    print("========================\n")
    
    if results_df.empty:
        return []

        
    # ---------------------------------
    # BHK FILTER
    # ---------------------------------
    if extracted_criteria["bhk"] and "bhk_type" in results_df.columns:

        target_bhk = extracted_criteria["bhk"]

        filtered_df = results_df[
            results_df["bhk_type"]
            .str.lower()
            .str.replace(" ", "")
            == target_bhk
        ]

        if not filtered_df.empty:
            results_df = filtered_df

        print("\n===== AFTER BHK FILTER =====")
        print("Rows:", len(results_df))

        cols = [
            c for c in ["id", "location", "city", "bhk_type"]
            if c in results_df.columns
        ]

        print(results_df[cols].head(10))

        print("============================\n")


    # ---------------------------------
    # LOCATION FILTER
    # ---------------------------------
    if extracted_criteria["location"]:

        target_location = (
            extracted_criteria["location"]
            .strip()
            .lower()
        )

        if target_location in [
            "mumbai",
            "thane",
            "navi mumbai",
            "palghar"
        ]:

            filtered_df = results_df[
                results_df["city"]
                .fillna("")
                .str.lower()
                == target_location
            ]

        else:

            filtered_df = results_df[
                results_df["location"]
                .fillna("")
                .str.lower()
                == target_location
            ]

        if not filtered_df.empty:
            results_df = filtered_df

        print("\n===== AFTER LOCATION FILTER =====")
        print("Rows:", len(results_df))

        cols = [
            c for c in ["id", "location", "city", "bhk_type"]
            if c in results_df.columns
        ]

        print(results_df[cols].head(10))

        print("=================================\n")

    out_cols = [c for c in SEARCH_RESULTS_WHITELIST if c in results_df.columns]
    return results_df[out_cols].head(limit).to_dict(orient="records")


def get_property_details(property_id: str) -> dict:
    """Perform a filtered drill-down isolation query matching a single asset ID."""
    target_id = str(property_id).strip()
    match_frame = GLOBAL_MASTER_DF[GLOBAL_MASTER_DF["id"].astype(str).str.strip() == target_id]
    
    if match_frame.empty:
        return {"error": f"No active recorded data elements matching individual asset key '{target_id}' could be isolated."}
        
    out_cols = [c for c in PROPERTY_DETAIL_WHITELIST if c in match_frame.columns]
    return match_frame[out_cols].to_dict(orient="records")[0]


def get_properties_by_ids(property_ids: list[str]) -> list[dict]:
    """Batch-loads and resolves multiple property detail records in a single round-trip call."""
    target_ids = [str(pid).strip() for pid in property_ids if pid]
    if not target_ids:
        return []
        
    match_frame = GLOBAL_MASTER_DF[GLOBAL_MASTER_DF["id"].astype(str).str.strip().isin(target_ids)]
    if match_frame.empty:
        return []
        
    out_cols = [c for c in PROPERTY_DETAIL_WHITELIST if c in match_frame.columns]
    return match_frame[out_cols].to_dict(orient="records")


# =====================================================================
# 2. PROPERTY EVALUATION TOOLS
# =====================================================================

def compare_properties(property_ids: list[str]) -> dict:
    """
    Compare multiple properties using MCP analysis.
    Sort properties by overall_score.
    Return the highest-scoring property as the winner
    along with the complete ranking list.
    """
    target_ids = [str(pid).strip() for pid in property_ids if pid]
    if len(target_ids) < 2:
        return {"error": "Analytical stacking comparison metrics require at least 2 distinct property IDs."}

    _, compare_df = run_mcp_comparison(target_ids)
    if compare_df.empty:
        return {"error": "Comparison pipeline execution returned an empty validation frame."}

    compare_df = compare_df.sort_values("overall_score", ascending=False)
    ranking_cols = ["id", "overall_score", "verdict", "comparison_reason"] #Keep only important columns.
    rankings = compare_df[ranking_cols].to_dict(orient="records") #Convert dataframe → list of dictionaries.
    
    return {
        "winner": rankings[0],
        "rankings": rankings
    }


def get_rental_analysis(property_ids: list[str]) -> list[dict]:
    """Extract micro-rental income generation metrics, estimates, and yields."""
    target_ids = [str(pid).strip() for pid in property_ids if pid]
    if not target_ids:
        return []
    return run_mcp_rental(target_ids).to_dict(orient="records")


def get_price_prediction(property_ids: list[str]) -> list[dict]:
    """Involves prediction core server logic routines to trace estimated variance metrics."""
    target_ids = [str(pid).strip() for pid in property_ids if pid]
    if not target_ids:
        return []
    return run_mcp_prediction(target_ids).to_dict(orient="records")


def get_negotiation_strategy(property_ids: list[str]) -> list[dict]:
    """Generates localized leverage power scores, target caps, and verbal counter strategies."""
    target_ids = [str(pid).strip() for pid in property_ids if pid]
    if not target_ids:
        return []
    return run_mcp_negotiation(target_ids).to_dict(orient="records")


def get_valuation_analysis(property_ids: list[str]) -> list[dict]:
    """Validates baseline distribution thresholds to identify fair market pricing parameters."""
    target_ids = [str(pid).strip() for pid in property_ids if pid]
    if not target_ids:
        return []
    return run_mcp_valuation(target_ids).to_dict(orient="records")


def get_investment_advice(property_ids: list[str]) -> list[dict]:
    """Runs structural profile matrices to chart risk metrics against investment horizons."""
    target_ids = [str(pid).strip() for pid in property_ids if pid]
    if not target_ids:
        return []
    return run_mcp_advisor(target_ids).to_dict(orient="records")


def clear_property_analysis_cache() -> dict:
    """Explicitly purges agent model enrichment caches while maintaining the immutable engine states intact."""
    clear_enrichment_cache()
    return {"status": "success", "message": "Global downstream property agent enrichment caches successfully flushed."}

# =====================================================================
# 3. N8N REPORT DELIVERY
# =====================================================================

def send_property_report(phone_number: str, report: str) -> dict:
    """
    Sends property report to n8n workflow.
    """

    webhook_url = "https://buzz123.app.n8n.cloud/webhook-test/98d3e0b7-9577-43ff-8f7a-a43956739ff9"

    response = requests.post(
        webhook_url,
        json={
            "phone": phone_number,
            "report": report
        },
        timeout=30
    )

    return {
        "status": "success",
        "status_code": response.status_code
    }




