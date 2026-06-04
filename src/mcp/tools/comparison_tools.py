# # =====================================================================
# # src/mcp/tools/comparison_tools.py
# # =====================================================================

# import json

# from src.services.mcp_comparison_service import (
#     run_mcp_comparison
# )


# def compare_properties(
#     property_ids: list[str]
# ) -> str:
#     """
#     Compare multiple properties and
#     return investment ranking.
#     """

#     if len(property_ids) < 2:
#         return json.dumps(
#             {
#                 "error":
#                 "Need at least 2 properties"
#             },
#             indent=2
#         )

#     raw_df, compare_df = run_mcp_comparison(
#         property_ids
#     )

#     if compare_df.empty:
#         return json.dumps(
#             {
#                 "error":
#                 "Comparison returned no results"
#             },
#             indent=2
#         )

#     compare_df = compare_df.sort_values(
#         "overall_score",
#         ascending=False
#     )

#     winner = compare_df.iloc[0]

#     result = {
#         "winner": {
#             "id": str(
#                 winner["id"]
#             ),
#             "overall_score": float(
#                 winner["overall_score"]
#             ),
#             "verdict": str(
#                 winner["verdict"]
#             ),
#             "comparison_reason": str(
#                 winner["comparison_reason"]
#             )
#         },
#         "rankings": compare_df[
#             [
#                 "id",
#                 "overall_score",
#                 "verdict",
#                 "comparison_reason"
#             ]
#         ].to_dict(
#             orient="records"
#         )
#     }

#     return json.dumps(
#         result,
#         indent=2,
#         default=str
#     )

#==========================================================================================

# # =====================================================================
# # src/mcp/tools/comparison_tools.py
# # =====================================================================

# import json
# from src.services.mcp_comparison_service import run_mcp_comparison


# def compare_properties(property_ids: list[str]) -> str:
#     """Compare multiple properties and return investment ranking."""

#     if len(property_ids) < 2:
#         return json.dumps({"error": "Need at least 2 properties"}, indent=2)

#     raw_df, compare_df = run_mcp_comparison(property_ids)

#     if compare_df.empty:
#         return json.dumps({"error": "Comparison returned no results"}, indent=2)

#     # Sort to determine rankings and the winner
#     compare_df = compare_df.sort_values("overall_score", ascending=False)
    
#     # Grab the relevant columns as a list of dicts
#     ranking_cols = ["id", "overall_score", "verdict", "comparison_reason"]
#     rankings = compare_df[ranking_cols].to_dict(orient="records")
    
#     # The winner is the first item in our sorted rankings list
#     result = {
#         "winner": rankings[0],
#         "rankings": rankings
#     }

#     return json.dumps(result, indent=2, default=str)