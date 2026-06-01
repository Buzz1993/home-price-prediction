# # ===============================
# # search_tool.py
# # ===============================
# # search_tool.py exposes the existing recommendation engine as an MCP tool. When an MCP client calls search_properties, the 
# # tool receives search filters, executes the existing run_search_pipeline function, retrieves the recommended properties, and returns the top results as a JSON response. 
# # This allows any MCP-compatible client to access the recommendation engine without directly depending on the project code.

# import pandas as pd

# from src.graph.workflow import search_graph
# from src.data.content_based_filtering import train


# df = pd.read_csv(
#     "data/cleaned/final_cleaned_rec_data.csv"
# )

# pipe, X_processed = train(df)


# def search_properties(
#     city: str,
#     bhk: int
# ):
#     """
#     Search properties using existing
#     recommendation engine.
#     """

#     filters = {
#         "city": city,
#         "bed": bhk
#     }

#     state = {

#         "df": df,

#         "X_processed": X_processed,

#         "filters": filters,

#         "intent": {},

#         "slider_weights": {},

#         "mode": "dynamic",

#         "selected_properties": pd.DataFrame(),

#         "recommendations": None,

#         "comparison_raw": None,

#         "comparison_result": None,

#         "explanation": None
#     }

#     result = search_graph.invoke(state)

#     recs = result["recommendations"]

#     return {
#         "input_count":
#             len(recs["input"]),

#         "recommendation_count":
#             len(recs["similar"]),

#         "properties":
#             recs["similar"][
#                 [
#                     "id",
#                     "project_name",
#                     "location",
#                     "price",
#                     "hybrid_score"
#                 ]
#             ].to_dict(
#                 orient="records"
#             )
#     }

#===========================================================

# =====================================================================
# src/mcp/tools/search_tool.py
# =====================================================================

from pathlib import Path
from src.utils.search_engine import RealEstateSearchEngine

# Resolve path locations relative to file depth block
ROOT_PATH = Path(__file__).resolve().parents[3]
DATA_FILE = ROOT_PATH / "data" / "cleaned" / "final_combined_mcp_data.csv"

# Spin up engine once into module state when application instances load
engine = RealEstateSearchEngine(DATA_FILE)

def execute_property_search(bhk: str = None, amenities: str = None, location: str = None, min_matches: int = 2):
    """
    Core tool provided to search agents and stream frontends.
    Requires a minimum number of distinct criteria fields to score a match.
    """
    print("search_tool get executed")
    search_criteria = {
        "bhk": bhk,
        "amenities": amenities,
        "location": location
    }
    
    # Strip away empty strings or unprovided filter states
    search_criteria = {k: v for k, v in search_criteria.items() if v and v.strip()}
    
    if not search_criteria:
        return {"error": "Search payload was completely empty."}
        
    results_df = engine.query(search_criteria, min_matches=min_matches)
    
    if results_df.empty:
        return {"message": "Zero matches satisfied your combined query parameters."}
        
    # Return top 10 rows for user evaluation or engine analysis context
    return results_df.head(10).to_dict(orient="records")