# #================================
# # mcp_comparison_service.py
# #================================

# from src.services.comparison_service import run_comparison
# from src.services.mcp_enrichment_service import enrich_properties
# from src.data.data_store import master_df


# def run_mcp_comparison(property_ids: list[str]):
#     """Load selected properties, enrich them, and run comparison."""

#     # Filter master dataframe for the requested IDs
#     selected_df = master_df[master_df["id"].isin(property_ids)].copy()

#     # Need at least 2 properties to perform a comparison
#     if len(selected_df) < 2:
#         return selected_df, selected_df

#     # Enrich and compare
#     enriched_df = enrich_properties(selected_df)
#     return run_comparison(enriched_df)