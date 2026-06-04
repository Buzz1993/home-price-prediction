# =====================================================================
# src/mcp/tools/property_tools.py
# =====================================================================

import json
from src.services.mcp_real_estate_service import (
    run_mcp_comparison,
    run_mcp_rental,
    run_mcp_prediction,
    run_mcp_negotiation,
    run_mcp_valuation,
    run_mcp_advisor
)


# =====================================================================
# 1. INVESTMENT COMPARISON & RANKING
# =====================================================================
def compare_properties(property_ids: list[str]) -> str:
    """Compare multiple properties and return investment ranking scores and verdicts."""
    if len(property_ids) < 2:
        return json.dumps({"error": "Need at least 2 properties for analytical comparison"}, indent=2)

    raw_df, compare_df = run_mcp_comparison(property_ids)

    if compare_df.empty:
        return json.dumps({"error": "Comparison returned no results"}, indent=2)

    # Sort to determine rankings and extract the clear winner
    compare_df = compare_df.sort_values("overall_score", ascending=False)
    
    ranking_cols = ["id", "overall_score", "verdict", "comparison_reason"]
    rankings = compare_df[ranking_cols].to_dict(orient="records")
    
    result = {
        "winner": rankings[0],
        "rankings": rankings
    }
    return json.dumps(result, indent=2, default=str)


# =====================================================================
# 2. RENTAL MATRIX ANALYTICS
# =====================================================================
def get_rental_analysis(property_ids: list[str]) -> str:
    """Run rental yield analysis, estimates, and demand metrics for given properties."""
    if not property_ids:
        return json.dumps({"error": "No properties provided for rental analysis"}, indent=2)

    rental_df = run_mcp_rental(property_ids)
    return json.dumps(
        rental_df.to_dict(orient="records"), 
        indent=2, 
        default=str
    )


# =====================================================================
# 3. ML PRICE PREDICTION MODEL
# =====================================================================
def get_price_prediction(property_ids: list[str]) -> str:
    """Invokes prediction engine models to forecast valuation pricing differences."""
    if not property_ids:
        return json.dumps({"error": "No properties provided for price prediction"}, indent=2)
        
    prediction_df = run_mcp_prediction(property_ids)
    return json.dumps(
        prediction_df.to_dict(orient="records"),
        indent=2,
        default=str
    )


# =====================================================================
# 4. NEGOTIATION STRATEGY GUIDE
# =====================================================================
def get_negotiation_strategy(property_ids: list[str]) -> str:
    """Generates localized buyer leverage power, target prices, and strategic talking points."""
    if not property_ids:
        return json.dumps({"error": "No properties available for strategy mapping"}, indent=2)
        
    negotiation_df = run_mcp_negotiation(property_ids)
    return json.dumps(
        negotiation_df.to_dict(orient="records"),
        indent=2,
        default=str
    )


# =====================================================================
# 5. BENCHMARK VALUATION ANALYTICS
# =====================================================================
def get_valuation_analysis(property_ids: list[str]) -> str:
    """Evaluates core market benchmarking thresholds to flag fair-market pricing deviations."""
    if not property_ids:
        return json.dumps({"error": "No target records isolated for pricing evaluation"}, indent=2)
        
    valuation_df = run_mcp_valuation(property_ids)
    return json.dumps(
        valuation_df.to_dict(orient="records"),
        indent=2,
        default=str
    )


# =====================================================================
# 6. PORTFOLIO INVESTMENT ADVISOR
# =====================================================================
def get_investment_advice(property_ids: list[str]) -> str:
    """Runs high-conviction decision scoring matrices to flag risks, positives, and buyer profiles."""
    if not property_ids:
        return json.dumps({"error": "No properties staged for investment advising"}, indent=2)
        
    advisor_df = run_mcp_advisor(property_ids)
    return json.dumps(
        advisor_df.to_dict(orient="records"),
        indent=2,
        default=str
    )