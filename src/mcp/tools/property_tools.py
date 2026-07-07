# ==============================
# src/mcp/tools/property_tools.py
# ==============================
"""
Unified Real Estate Analytical Tools Layer.
Enforces whitelists across search results and batch detail projections to maintain token efficiency.
"""
import pandas as pd
import requests
from src.core.search_registry import (
    SEARCH_STATE,
    GLOBAL_MASTER_DF,
    CACHED_SEARCH_METADATA,
    query,
)
from src.services.mcp_real_estate_service import (
    run_mcp_comparison,
    run_mcp_rental,
    run_mcp_prediction,
    run_mcp_negotiation,
    run_mcp_valuation,
    run_mcp_advisor,
    clear_enrichment_cache
)

from src.llm.deepseek_client import ask_deepseek

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
    """
    This function searches the property dataset using the filters
    extracted from the user's query (BHK, amenities, and location).
    
    It:
    1. Builds the search criteria from the extracted filters.
    2. Calls the BM25 search engine (query()) to retrieve the
       best matching properties.
    3. Applies exact BHK filtering so only the requested BHK
       properties remain.
    4. Applies exact location or city filtering to remove
       unrelated properties.
    5. Keeps only the required columns for the UI.
    6. Converts the DataFrame into a list of dictionaries.
    
    Returns:
    [
        {
            "id": 101,
            "price": 9500000,
            "bhk_type": "2bhk",
            "location": "Thane",
            "city": "Thane",
            "search_score": 0.91
        },
        {
            "id": 205,
            "price": 8700000,
            "bhk_type": "2bhk",
            "location": "Thane",
            "city": "Thane",
            "search_score": 0.89
        },
        ...
    ].
    """
    print("☑️ search_properties executed")
    
    # ==================================================
    # BUILD SEARCH CRITERIA
    # ==================================================
    extracted_criteria = {
        "bhk": f"{bhk.strip().lower().replace(' ', '')}" if bhk else None,
        "amenities": amenities,
        "location": location
    }

    # ==================================================
    # VALIDATE SEARCH INPUT
    # ==================================================
    # Return an empty list if no search filters are provided.
    
    active_criteria = {k: v for k, v in extracted_criteria.items() if v}
    if not active_criteria:
        return []
        
    # ==================================================
    # EXECUTE BM25 SEARCH
    # ==================================================
    # Search the property dataset using the extracted criteria.    
    determined_min_matches = 1 if (not amenities or not location) else 2
    results_df = query(SEARCH_STATE, extracted_criteria, min_matches=determined_min_matches)

    print("\n===== QUERY OUTPUT =====")

    print("Criteria:", extracted_criteria)
    print("Rows Returned:", len(results_df))

    if not results_df.empty:
        cols = [c for c in ["id", "location", "city", "bhk_type"] if c in results_df.columns]
        print(results_df[cols].head(10))

    print("========================\n")
    

    # ==================================================
    # EMPTY RESULT CHECK
    # ==================================================
    # Stop if no matching properties are found.
    if results_df.empty:
        return []

        
    # ==================================================
    # BHK FILTER
    # ==================================================
    # Keep only properties matching the requested BHK.
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


    # ==================================================
    # LOCATION FILTER
    # ==================================================
    # Filter by city or locality depending on the user's input.
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


    # ==================================================
    # RETURN SEARCH RESULTS
    # ==================================================
    # Keep only the required columns and return the
    # matching properties as a list of dictionaries.
    out_cols = [c for c in SEARCH_RESULTS_WHITELIST if c in results_df.columns]
    return results_df[out_cols].head(limit).to_dict(orient="records")


def get_property_details(property_id: str) -> dict:
    """Perform a filtered drill-down isolation query matching a single asset ID."""
    print("☑️ get_property_details executed")
    target_id = str(property_id).strip()
    match_frame = GLOBAL_MASTER_DF[GLOBAL_MASTER_DF["id"].astype(str).str.strip() == target_id]
    
    if match_frame.empty:
        return {"error": f"No active recorded data elements matching individual asset key '{target_id}' could be isolated."}
        
    out_cols = [c for c in PROPERTY_DETAIL_WHITELIST if c in match_frame.columns]
    return match_frame[out_cols].to_dict(orient="records")[0]


def get_properties_by_ids(property_ids: list[str]) -> list[dict]:
    """Batch-loads and resolves multiple property detail records in a single round-trip call."""
    print("☑️ get_properties_by_ids executed")
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

def compare_properties(property_ids: list[str]) -> dict: # Receives the selected property IDs from the tray.
    """
    Compare multiple properties using MCP analysis.
    Sort properties by overall_score.
    Return the highest-scoring property as the winner
    along with the complete ranking list.
    """
    print("☑️ compare_properties executed")
    print("=" * 50)
    print("Raw property_ids received:", property_ids)
    print("=" * 50)

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
    print("☑️ get_rental_analysis executed")
    target_ids = [str(pid).strip() for pid in property_ids if pid]
    if not target_ids:
        return []
    return run_mcp_rental(target_ids).to_dict(orient="records")


def get_price_prediction(property_ids: list[str]) -> list[dict]:
    """Involves prediction core server logic routines to trace estimated variance metrics."""
    print("☑️ get_price_prediction executed")
    target_ids = [str(pid).strip() for pid in property_ids if pid]
    if not target_ids:
        return []
    return run_mcp_prediction(target_ids).to_dict(orient="records")


def get_negotiation_strategy(property_ids: list[str]) -> list[dict]:
    """Generates localized leverage power scores, target caps, and verbal counter strategies."""
    print("☑️ get_negotiation_strategy executed")
    target_ids = [str(pid).strip() for pid in property_ids if pid]
    if not target_ids:
        return []
    return run_mcp_negotiation(target_ids).to_dict(orient="records")


def get_valuation_analysis(property_ids: list[str]) -> list[dict]:
    """Validates baseline distribution thresholds to identify fair market pricing parameters."""
    print("☑️ get_valuation_analysis executed")
    target_ids = [str(pid).strip() for pid in property_ids if pid]
    if not target_ids:
        return []
    return run_mcp_valuation(target_ids).to_dict(orient="records")


def get_investment_advice(property_ids: list[str]) -> list[dict]:
    """Runs structural profile matrices to chart risk metrics against investment horizons."""
    print("☑️ get_investment_advice executed")
    target_ids = [str(pid).strip() for pid in property_ids if pid]
    if not target_ids:
        return []
    return run_mcp_advisor(target_ids).to_dict(orient="records")


def clear_property_analysis_cache() -> dict:
    """Explicitly purges agent model enrichment caches while maintaining the immutable engine states intact."""
    print("☑️ clear_property_analysis_cache executed")
    clear_enrichment_cache()
    return {"status": "success", "message": "Global downstream property agent enrichment caches successfully flushed."}



# =====================================================
# REPORT GENERATION
# =====================================================

def create_property_report(property_ids: list[str]) -> str:
    """
    Generate a Markdown property report for the
    selected properties.

    The report is returned as plain text and can be:

    • previewed in the frontend
    • downloaded
    • sent to n8n via send_property_report()
    """

    if not property_ids:
        raise ValueError("At least one property is required.")

    # -------------------------------------------------
    # Load property details
    # -------------------------------------------------

    properties = get_properties_by_ids(property_ids)

    if not properties:
        raise ValueError("No matching properties found.")

    # -------------------------------------------------
    # Build context for the LLM
    # -------------------------------------------------

    property_context = ""

    for i, property_data in enumerate(properties, start=1):

        property_context += f"""
Property {i}

ID: {property_data.get("id")}
Project: {property_data.get("project_name")}
Builder: {property_data.get("builder")}
Location: {property_data.get("location")}
Price: {property_data.get("price")}
Area: {property_data.get("area")}
BHK: {property_data.get("bhk_type")}
Amenities:
{property_data.get("amenities_mcp")}
Features:
{property_data.get("features_mcp")}

"""

    # -------------------------------------------------
    # Prompt
    # -------------------------------------------------

    prompt = f"""
You are EstateMind AI.

Generate a professional real estate report in Markdown.

Include:

# Executive Summary

# Property Analysis

# Strengths

# Weaknesses

# Investment Potential

# Recommendation

Properties

{property_context}

Use Indian Rupee (₹).

Return Markdown only.
"""

    # -------------------------------------------------
    # Generate report
    # -------------------------------------------------

    report = ask_deepseek(prompt)

    if not report:
        raise ValueError("Failed to generate report.")

    return report


# =====================================================================
# 3. N8N REPORT DELIVERY
# =====================================================================

def send_property_report(phone_number: str, report: str) -> dict:
    """
    Sends property report to n8n workflow.
    """
    print("☑️ send_property_report executed")
    webhook_url = "https://bush123.app.n8n.cloud/webhook-test/98d3e0b7-9577-43ff-8f7a-a43956739ff9"

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




