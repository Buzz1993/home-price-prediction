# """
# Centralized Search Registry.
# Completely decoupled module containing zero local dependencies to eliminate circular paths.
# Exposes root-level search schema mappings for reusability across data layers and testing suites.
# """
# import json
# from pathlib import Path
# import pandas as pd
# from rank_bm25 import BM25Okapi
# from src.data.data_store import master_df
# from src.utils.search_metadata import clean_and_split

# ROOT_DIR = Path(__file__).resolve().parents[2]
# METADATA_PATH = ROOT_DIR / "data" / "cleaned" / "search_metadata.json"

# # --- GLOBAL REUSABLE CONFIGURATION SCHEMAS ---
# SEARCH_SCHEMA = {
#     "bhk": ["bhk_type", "extra_rooms"],  
#     "property_details": ["property_type", "status", "furnish", "construction", "ownership"],
#     "specifications": ["facing", "flooring", "builder"],
#     "amenities": ["amenities_mcp", "features_mcp", "amenities_text", "overlooking"],
#     "location": ["location", "city", "transportation_hubs_clean", "project_name", "nearest_mcp"]
# }

# def _tokenize_string(text, column_name: str) -> list[str]:
#     if pd.isna(text):
#         return []
#     return [
#         word 
#         for phrase in clean_and_split(str(text), column_name) 
#         for word in str(phrase).split()
#     ]

# def _precompute_bm25_indices(df: pd.DataFrame) -> dict:
#     print("🧠 Slicing and compiling tokenized BM25 matrices...")
#     clean_df = df.reset_index(drop=True)
#     computed_indices = {}

#     for category, columns in SEARCH_SCHEMA.items():
#         valid_cols = [col for col in columns if col in clean_df.columns]
#         corpus = [
#             [token for col in valid_cols for token in _tokenize_string(row[col], col)]
#             for _, row in clean_df.iterrows()
#         ]
#         computed_indices[category] = BM25Okapi(corpus)
        
#     print("🚀 Decoupled BM25 Matrix Search Indices generated successfully!")
#     return {"df": clean_df, "indexes": computed_indices}

# def _load_static_metadata_cache() -> dict:
#     try:
#         if METADATA_PATH.exists():
#             with open(METADATA_PATH, "r", encoding="utf-8") as f:
#                 print("📊 Search metadata lookup file loaded into memory register cache.")
#                 return json.load(f)
#         return {"error": "Metadata tracking file index missing from persistent storage."}
#     except Exception as e:
#         return {"error": f"Failed memory-mapping core schema metadata profiles: {str(e)}"}

# # --- IN-MEMORY IMMUTABLE CACHE REGISTRIES ---
# print("📌 Bootstrapping Decoupled Global Asset Memory Registry...")
# GLOBAL_MASTER_DF = master_df
# SEARCH_STATE = _precompute_bm25_indices(GLOBAL_MASTER_DF)
# CACHED_SEARCH_METADATA = _load_static_metadata_cache()

#==========================================================================================================================================================================
# # ===============================
# # search_registry.py
# # ===============================
# """
# Centralized Search Registry.
# Completely decoupled module containing zero local dependencies to eliminate circular paths.
# Exposes root-level search schema mappings for reusability across data layers and testing suites.
# """
# import json
# from pathlib import Path
# import sys
# import pandas as pd
# from rank_bm25 import BM25Okapi
# from src.data.data_store import master_df
# from src.utils.search_metadata import clean_and_split

# ROOT_DIR = Path(__file__).resolve().parents[2]
# METADATA_PATH = ROOT_DIR / "data" / "cleaned" / "search_metadata.json" #this we get from search_metadata.py

# # --- GLOBAL REUSABLE CONFIGURATION SCHEMAS ---
# SEARCH_SCHEMA = {
#     "bhk": ["bhk_type", "extra_rooms"],  
#     "property_details": ["property_type", "status", "furnish", "construction", "ownership"],
#     "specifications": ["facing", "flooring", "builder"],
#     "amenities": ["amenities_mcp", "features_mcp", "amenities_text", "overlooking"],
#     "location": ["location", "city", "transportation_hubs_clean", "project_name", "nearest_mcp"]
# }




# #each row gets its own list of tokens
# #"Swimming Pool, CCTV Camera" → ["swimming", "pool", "cctv", "camera"]
# #"Goregaon East" → ["goregaon", "east"]
# def _tokenize_string(text, column_name: str) -> list[str]:
#     if pd.isna(text):
#         return []
#     return [
#         word 
#         for phrase in clean_and_split(str(text), column_name) 
#         for word in str(phrase).split()
#     ]



# # _precompute_bm25_indices() builds the BM25 indexes by calling _tokenize_string() for every relevant cell in every row.
# # Builds all the BM25 search indexes once when the application starts.
# # SEARCH_STATE conceptually looks like this (assuming there are only 2 rows):
# '''
# SEARCH_STATE = {

#     "df": clean_df,

#     "indexes": {

#         "location": BM25Okapi([
#             ["goregaon", "east"],
#             ["andheri", "west"]
#         ]),

#         "amenities": BM25Okapi([
#             ["cctv", "camera", "swimming", "pool"],
#             ["gymnasium", "lift"]
#         ]),

#         "bhk": BM25Okapi([
#             ["2", "bhk"],
#             ["3", "bhk"]
#         ])
#     }
# }
# '''
# # Note:
# # The token lists above are only used to BUILD the BM25 indexes.
# # SEARCH_STATE actually stores BM25Okapi objects internally, like this:
# '''
# {
#     "location": <BM25Okapi object>,
#     "amenities": <BM25Okapi object>,
#     "bhk": <BM25Okapi object>
# }
# '''
# def _precompute_bm25_indices(df: pd.DataFrame) -> dict:
#     print(
#         "🧠 Slicing and compiling tokenized BM25 matrices...",
#         file=sys.stderr
#     )
#     clean_df = df.reset_index(drop=True)
#     computed_indices = {}

#     for category, columns in SEARCH_SCHEMA.items():
#         valid_cols = [col for col in columns if col in clean_df.columns]
#         corpus = [
#             [token for col in valid_cols for token in _tokenize_string(row[col], col)]
#             for _, row in clean_df.iterrows()
#         ]
#         computed_indices[category] = BM25Okapi(corpus)
        
#     print(
#         "🚀 Decoupled BM25 Matrix Search Indices generated successfully!",
#         file=sys.stderr
#     )
#     return {"df": clean_df, "indexes": computed_indices}




# #This function loads the search_metadata.json(this we get from search_metadata.py) file into memory once when the application starts 
# #so it can be accessed quickly without reading the file again and again.
# def _load_static_metadata_cache() -> dict:
#     try:
#         if METADATA_PATH.exists():
#             with open(METADATA_PATH, "r", encoding="utf-8") as f:
#                 print(
#                     "📊 Search metadata lookup file loaded into memory register cache.",
#                     file=sys.stderr
#                 )
#                 return json.load(f)
#         return {"error": "Metadata tracking file index missing from persistent storage."}
#     except Exception as e:
#         return {"error": f"Failed memory-mapping core schema metadata profiles: {str(e)}"}

# # --- IN-MEMORY IMMUTABLE CACHE REGISTRIES ---
# print(
#     "📌 Bootstrapping Decoupled Global Asset Memory Registry...",
#     file=sys.stderr
# )
# GLOBAL_MASTER_DF = master_df
# SEARCH_STATE = _precompute_bm25_indices(GLOBAL_MASTER_DF)
# CACHED_SEARCH_METADATA = _load_static_metadata_cache()



# # Code understanding notes
# # GLOBAL_MASTER_DF
# # ----------------
# # Stores the complete property dataset (actual property database).
# #
# # Example:
# # --------------------------------------------------------
# # id     location     amenities                 price
# # P101   Goregaon     CCTV Camera, Pool         2.1 Cr
# # P102   Andheri      Gymnasium, Lift           3.0 Cr
# # --------------------------------------------------------


# # CACHED_SEARCH_METADATA
# # ----------------------
# # Stores the unique cleaned values from selected searchable columns
# # (one combined list per column for the entire dataset).
# #
# # Example:
# # {
# #     "location": [
# #         "Goregaon",
# #         "Andheri",
# #         "Powai"
# #     ],
# #
# #     "amenities_mcp": [
# #         "cctv camera",
# #         "swimming pool",
# #         "gymnasium"
# #     ],
# #
# #     "builder": [
# #         "Lodha",
# #         "Godrej"
# #     ]
# # }
# #
# # Used for:
# # - Detecting locations in the user's query.
# # - Detecting amenities in the user's query.
# # - Fallback parsing if the LLM misses a value.


# # SEARCH_STATE
# # ------------
# # Stores the BM25 search indexes built from the tokenized
# # searchable content of every row in the dataset.
# #
# # Example dataset:
# # --------------------------------------------------------
# # id     location     amenities
# # P101   Goregaon     CCTV Camera, Swimming Pool
# # P102   Andheri      Gymnasium, Lift
# # --------------------------------------------------------
# #
# # After tokenization:
# #
# # Row 1
# # ["goregaon", "cctv", "camera", "swimming", "pool"]
# #
# # Row 2
# # ["andheri", "gymnasium", "lift"]
# #
# # If the user searches:
# #
# #     "goregaon cctv"
# #
# # BM25 compares the search query with the tokenized searchable
# # content of every row, assigns a relevance score to each row,
# # and returns the best matching properties in ranked order.



# # Why do we use BM25?
# #
# # A normal keyword search only checks whether a property contains the searched words.
# #
# # Example:
# # User searches: "cctv pool"
# #
# # Dataset:
# # --------------------------------------------------
# # Row 1 : CCTV
# # Row 2 : CCTV, Swimming Pool
# # Row 3 : Gym
# # --------------------------------------------------
# #
# # Normal search result:
# # ✓ Row 1 matches ("cctv")
# # ✓ Row 2 matches ("cctv" and "pool")
# # ✗ Row 3 doesn't match
# #
# # Problem:
# # A normal search cannot determine which matching row is more relevant.
# #
# # BM25 solves this by assigning a relevance score to each row.
# #
# # BM25 result:
# # --------------------------------------------------
# # Row 2 : CCTV, Swimming Pool   → Score = 9.8
# # Row 1 : CCTV                  → Score = 5.1
# # Row 3 : Gym                   → Score = 0.0
# # --------------------------------------------------
# #
# # Therefore, BM25 ranks Row 2 above Row 1 because it matches more of
# # the user's search query and is more relevant.


#================================================================================================================================================================================================

# ===============================
# search_registry.py
# ===============================
"""
Centralized Search Registry.
Completely decoupled module containing zero local dependencies to eliminate circular paths.
Exposes root-level search schema mappings for reusability across data layers and testing suites.
"""
import json
from pathlib import Path
import sys
import pandas as pd
from rank_bm25 import BM25Okapi
from src.data.data_store import master_df
from src.utils.search_metadata import clean_and_split

ROOT_DIR = Path(__file__).resolve().parents[2]
METADATA_PATH = ROOT_DIR / "data" / "cleaned" / "search_metadata.json" #this we get from search_metadata.py

# --- GLOBAL REUSABLE CONFIGURATION SCHEMAS ---
SEARCH_SCHEMA = {
    "bhk": ["bhk_type", "extra_rooms"],  
    "property_details": ["property_type", "status", "furnish", "construction", "ownership"],
    "specifications": ["facing", "flooring", "builder"],
    "amenities": ["amenities_mcp", "features_mcp", "amenities_text", "overlooking"],
    "location": ["location", "city", "transportation_hubs_clean", "project_name", "nearest_mcp"]
}




#explanation for all codes already given below at the end

#each row gets its own list of tokens
#"Swimming Pool, CCTV Camera" → ["swimming", "pool", "cctv", "camera"]
#"Goregaon East" → ["goregaon", "east"]
def _tokenize_string(text, column_name: str) -> list[str]:
    if pd.isna(text):
        return []
    return [
        word 
        for phrase in clean_and_split(str(text), column_name) 
        for word in str(phrase).split()
    ]



# _precompute_bm25_indices() builds the BM25 indexes by calling _tokenize_string() for every relevant cell in every row.
# Builds all the BM25 search indexes once when the application starts.
# SEARCH_STATE conceptually looks like this (assuming there are only 2 rows):
'''
SEARCH_STATE = {

    "df": clean_df,

    "indexes": {

        "location": BM25Okapi([
            ["goregaon", "east"],
            ["andheri", "west"]
        ]),

        "amenities": BM25Okapi([
            ["cctv", "camera", "swimming", "pool"],
            ["gymnasium", "lift"]
        ]),

        "bhk": BM25Okapi([
            ["2", "bhk"],
            ["3", "bhk"]
        ])
    }
}
'''
# Note:
# The token lists above are only used to BUILD the BM25 indexes.
# SEARCH_STATE actually stores BM25Okapi objects internally, like this:
'''
{
    "location": <BM25Okapi object>,
    "amenities": <BM25Okapi object>,
    "bhk": <BM25Okapi object>
}
'''
def _precompute_bm25_indices(df: pd.DataFrame) -> dict:
    print(
        "🧠 Slicing and compiling tokenized BM25 matrices...",
        file=sys.stderr
    )
    clean_df = df.reset_index(drop=True)
    computed_indices = {}

    for category, columns in SEARCH_SCHEMA.items():
        valid_cols = [col for col in columns if col in clean_df.columns]
        corpus = [
            [token for col in valid_cols for token in _tokenize_string(row[col], col)]
            for _, row in clean_df.iterrows()
        ]
        computed_indices[category] = BM25Okapi(corpus)
        
    print(
        "🚀 Decoupled BM25 Matrix Search Indices generated successfully!",
        file=sys.stderr
    )
    return {"df": clean_df, "indexes": computed_indices}




# This function loads the search_metadata.json file (generated by search_metadata.py)
# into memory when the application starts, so the metadata can be accessed quickly
# without reading the JSON file from disk every time it is needed.
def _load_static_metadata_cache() -> dict:
    try:
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                print(
                    "📊 Search metadata lookup file loaded into memory register cache.",
                    file=sys.stderr
                )
                return json.load(f)
        return {"error": "Metadata tracking file index missing from persistent storage."}
    except Exception as e:
        return {"error": f"Failed memory-mapping core schema metadata profiles: {str(e)}"}

# --- IN-MEMORY IMMUTABLE CACHE REGISTRIES ---
print(
    "📌 Bootstrapping Decoupled Global Asset Memory Registry...",
    file=sys.stderr
)
GLOBAL_MASTER_DF = master_df
SEARCH_STATE = _precompute_bm25_indices(GLOBAL_MASTER_DF)
CACHED_SEARCH_METADATA = _load_static_metadata_cache()




# This function searches the BM25 indexes, scores every property against the
# user's search criteria, ranks the matching properties, and returns the
# matching properties as a DataFrame.
def query(engine_state: dict, search_criteria: dict, min_matches: int = 2) -> pd.DataFrame:
    """Scores properties against criteria and filters by minimum category match rules."""
    df = engine_state["df"] #get the dataset and BM25 indexes from _precompute_bm25_indices
    indexes = engine_state["indexes"]
    
    num_properties = len(df) # Count how many properties are in the cleaned dataset.
    match_counts = [0] * num_properties # Track how many search criteria each property matches. Example (4 properties): then becoz of 4 we get 0's as [0, 0 ,0, 0]
    total_scores = [0.0] * num_properties # Track the total BM25 score for each property. Example: total_scores = [0.0, 0.0, 0.0, 0.0]

    for category, query_str in search_criteria.items():
        if not query_str or category not in indexes:
            continue
        
        # Convert the search text into tokens.
        # Example:
        # "Swimming Pool" -> ["swimming", "pool"]
        query_tokens = _tokenize_string(query_str, "features_mcp")
        if not query_tokens: # Skip if no valid tokens were found.
            continue
            
        scores = indexes[category].get_scores(query_tokens) # BM25 returns one relevance score for every property.
        
        # Update each property's match count and total BM25 score.
        for idx, score in enumerate(scores):
            if score > 0.0:
                match_counts[idx] += 1
                total_scores[idx] += score

    # Keep only properties matching at least 'min_matches' search categories.
    matched_records = [
        {"row_idx": idx, "criteria_matched": match_counts[idx], "search_score": total_scores[idx]}
        for idx in range(num_properties)
        if match_counts[idx] >= min_matches
    ]

    # Return an empty DataFrame if no properties matched.    
    if not matched_records:
        return pd.DataFrame()

    # Sort results by:
    # 1. Number of matched search categories (highest first)
    # 2. Total BM25 score (highest first)
    results_df = pd.DataFrame(matched_records).sort_values(
        by=["criteria_matched", "search_score"], 
        ascending=[False, False]
    )
    
    final_df = results_df.merge(df, left_on="row_idx", right_index=True) # Merge the search scores with the original property dataset.
    return final_df.drop(columns=["row_idx"]) # Remove the temporary row index column and return the final ranked properties.






# Code understanding notes
# GLOBAL_MASTER_DF
# ----------------
# Stores the complete property dataset (actual property database).
#
# Example:
# --------------------------------------------------------
# id     location     amenities                 price
# P101   Goregaon     CCTV Camera, Pool         2.1 Cr
# P102   Andheri      Gymnasium, Lift           3.0 Cr
# --------------------------------------------------------


# CACHED_SEARCH_METADATA
# ----------------------
# Stores the unique cleaned values from selected searchable columns
# (one combined list per column for the entire dataset).
#
# Example:
# {
#     "location": [
#         "Goregaon",
#         "Andheri",
#         "Powai"
#     ],
#
#     "amenities_mcp": [
#         "cctv camera",
#         "swimming pool",
#         "gymnasium"
#     ],
#
#     "builder": [
#         "Lodha",
#         "Godrej"
#     ]
# }
#
# Used for:
# - Detecting locations in the user's query.
# - Detecting amenities in the user's query.
# - Fallback parsing if the LLM misses a value.


# SEARCH_STATE
# ------------
# Stores the BM25 search indexes built from the tokenized
# searchable content of every row in the dataset.
#
# Example dataset:
# --------------------------------------------------------
# id     location     amenities
# P101   Goregaon     CCTV Camera, Swimming Pool
# P102   Andheri      Gymnasium, Lift
# --------------------------------------------------------
#
# After tokenization:
#
# Row 1
# ["goregaon", "cctv", "camera", "swimming", "pool"]
#
# Row 2
# ["andheri", "gymnasium", "lift"]
#
# If the user searches:
#
#     "goregaon cctv"
#
# BM25 compares the search query with the tokenized searchable
# content of every row, assigns a relevance score to each row,
# and returns the best matching properties in ranked order.



# Why do we use BM25?
#
# A normal keyword search only checks whether a property contains the searched words.
#
# Example:
# User searches: "cctv pool"
#
# Dataset:
# --------------------------------------------------
# Row 1 : CCTV
# Row 2 : CCTV, Swimming Pool
# Row 3 : Gym
# --------------------------------------------------
#
# Normal search result:
# ✓ Row 1 matches ("cctv")
# ✓ Row 2 matches ("cctv" and "pool")
# ✗ Row 3 doesn't match
#
# Problem:
# A normal search cannot determine which matching row is more relevant.
#
# BM25 solves this by assigning a relevance score to each row.
#
# BM25 result:
# --------------------------------------------------
# Row 2 : CCTV, Swimming Pool   → Score = 9.8
# Row 1 : CCTV                  → Score = 5.1
# Row 3 : Gym                   → Score = 0.0
# --------------------------------------------------
#
# Therefore, BM25 ranks Row 2 above Row 1 because it matches more of
# the user's search query and is more relevant.



#explanation for query function
# What does query() do?
#
# This function searches the BM25 indexes using the user's search criteria,
# scores every property, keeps the best matches, and returns them as a DataFrame.
#
# Example:
#
# User search criteria:
#
# {
#     "bhk": "2bhk",
#     "location": "goregaon",
#     "amenities": "cctv"
# }
#
#
# Property Dataset:
# -------------------------------------------------------------------------
# Row   Property   BHK    Location     Amenities
# -------------------------------------------------------------------------
# 0     P101       2bhk   Goregaon     CCTV Camera
# 1     P102       2bhk   Andheri      Gymnasium
# 2     P103       3bhk   Goregaon     Swimming Pool
# 3     P104       2bhk   Goregaon     CCTV Camera, Swimming Pool
# -------------------------------------------------------------------------
#
#
# Step 1:
# BM25 searches every category separately.
#2bhk is in 1,2,and 4th and hence for 3rd we get 0.0, similarly for others
# -------------------------------------------------------------------------
# Search Criteria     BM25 Scores Returned
# -------------------------------------------------------------------------
# "2bhk"          -> [3.2, 3.0, 0.0, 3.5]
# "goregaon"      -> [5.4, 0.0, 5.0, 5.5]
# "cctv"          -> [6.2, 0.0, 0.0, 6.4]
# -------------------------------------------------------------------------
#
#
# Step 2:
# The function combines all category scores.
#for property P101 we have all this "2bhk", "goregaon", "cctv"  hence Matched Criteria become 3 for P101
# -------------------------------------------------------------------------
# Property   Matched Criteria     Total BM25 Score
# -------------------------------------------------------------------------
# P101              3                  14.8
# P102              1                   3.0
# P103              1                   5.0
# P104              3                  15.4
# -------------------------------------------------------------------------
#
#
# Step 3:
# Only properties matching at least 'min_matches' categories are kept.
#
# Example:
#
# min_matches = 2
#
# ✓ P101 kept
# ✗ P102 removed
# ✗ P103 removed
# ✓ P104 kept
#
#
# Step 4:
# Remaining properties are sorted by:
#
# 1. Highest number of matched criteria.
# 2. Highest total BM25 score.
#
#
# Final Result:
# -------------------------------------------------------------------------
# Property   Matched Criteria     BM25 Score
# -------------------------------------------------------------------------
# P104              3                15.4
# P101              3                14.8
# -------------------------------------------------------------------------
#
#
# Finally, the function merges these search scores with the original
# property dataset and returns the complete property information as
# a ranked DataFrame.