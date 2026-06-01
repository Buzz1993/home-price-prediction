# # src/mcp/test_search.py

# # import asyncio
# # from fastmcp import Client

# # async def main():

# #     async with Client(
# #         "src/mcp/server.py"
# #     ) as client:

# #         result = await client.call_tool(
# #             "search_properties",
# #             {
# #                 "filters": {
# #                     "city": "thane"
# #                 }
# #             }
# #         )

# #         print(result)

# # asyncio.run(main())

# import asyncio
# from fastmcp import Client

# async def main():

#     async with Client("src/mcp/server.py") as client:

#         result = await client.call_tool(
#             "search_properties",
#             {
#                 "filters": {
#                     "city": "thane"
#                 }
#             }
#         )

#         print("Count:", result.data["count"])

#         print("\nFirst 3 Properties:\n")

#         for prop in result.data["properties"][:3]:
#             print(
#                 prop.get("id"),
#                 "|",
#                 prop.get("city"),
#                 "|",
#                 prop.get("PRICE")
#             )

# asyncio.run(main())

#==========================================================================================

# =====================================================================
# src/mcp/test_search.py
# =====================================================================

import json
from pathlib import Path
import sys

# Ensure the root 'src' directory is in the Python path for clean imports
root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.mcp.tools.search_tool import execute_property_search


def run_mcp_search_tests():
    print("🧪 ===================================================")
    print("🧪 RUNNING MCP SEARCH ENGINE INTEGRATION TESTS")
    print("🧪 ===================================================\n")
    
    # 1. Path Verification Check
    data_file = root_path / "data" / "cleaned" / "final_combined_mcp_data.csv"
    print(f"📊 Target Dataset Path: {data_file}")
    print(f"📁 Target Dataset Exists: {data_file.exists()}")
    
    if not data_file.exists():
        print("❌ Error: 'final_combined_mcp_data.csv' not found at the expected location.")
        print("   Please run your data cleaning pipeline first to generate this file!")
        return

    # -----------------------------------------------------------------
    # Test Case 1: Multi-field intent evaluation with strict threshold gate
    # -----------------------------------------------------------------
    print("\n========================================================")
    print("🎯 TEST 1: Complex Multi-Field Matching (min_matches=2)")
    print("👉 Prompt: 2bhk property with cctv near goregaon railway station")
    print("========================================================")
    
    test_criteria = {
        "bhk": "2bhk",
        "amenities": "cctv kids playroom",
        "location": "goregaon railway station"
    }

    print("🛰️ Dispatching structured search parameters to search_tool...")
    results = execute_property_search(
        bhk=test_criteria["bhk"],
        amenities=test_criteria["amenities"],
        location=test_criteria["location"],
        min_matches=2
    )

    # Validate Response Structural Outputs
    if isinstance(results, dict) and ("error" in results or "message" in results):
        print(f"⚠️ Search Layer Notice: {results.get('error') or results.get('message')}")
    else:
        print(f"✅ Success! Retrieved {len(results)} properties matching >= 2 conditions.\n")
        
        # Display the top 3 best matching properties
        for idx, item in enumerate(results[:3]):
            print(f"🏠 [Rank Match #{idx+1}] ID: {item.get('id')}")
            print(f"   • BHK Found:   {item.get('bhk_type')}")
            print(f"   • Locality:    {item.get('location')} ({item.get('city')})")
            print(f"   • Trans Hubs:  {item.get('transportation_hubs_clean')}")
            print(f"   • Amenities:   {item.get('amenities_mcp')}")
            print(f"   • Features:    {item.get('features_mcp')}")
            print(f"   • BM25 Score:  {item.get('search_score')}")
            print(f"   • Price (Cr):  {item.get('price')}")
            print("-" * 40)

    # -----------------------------------------------------------------
    # Test Case 2: Lexical token matching verification (Camera -> CCTV)
    # -----------------------------------------------------------------
    print("\n========================================================")
    print("🎯 TEST 2: Token Overlap Fallback Gateway Validation")
    print("👉 Prompt: 'camera surveillance' (Looking for CCTV matches)")
    print("========================================================")
    
    results_fuzzy = execute_property_search(
        amenities="camera surveillance",
        min_matches=1
    )

    if isinstance(results_fuzzy, dict) and ("error" in results_fuzzy or "message" in results_fuzzy):
        print(f"❌ Failed: Could not parse token matches. Error: {results_fuzzy}")
    else:
        print(f"✅ Success! Extracted {len(results_fuzzy)} rows via token overlap.")
        # Sniff the text contents of the first result to make sure 'cctv' was captured
        top_match_features = results_fuzzy[0].get('features_mcp', '')
        print(f"   • Top Match Raw Features String: {top_match_features}")
        
        if "cctv" in str(top_match_features).lower():
            print("\n🔥 Test Passed: System correctly linked user term 'camera' to data token 'cctv' via 'surveillance' word matching!")
        else:
            print("\n⚠️ Warning: Shared token matched, but 'cctv' wasn't prominently displayed in the top match row.")


if __name__ == "__main__":
    run_mcp_search_tests()