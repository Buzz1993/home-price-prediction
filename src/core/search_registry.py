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
METADATA_PATH = ROOT_DIR / "data" / "cleaned" / "search_metadata.json"

# --- GLOBAL REUSABLE CONFIGURATION SCHEMAS ---
SEARCH_SCHEMA = {
    "bhk": ["bhk_type", "extra_rooms"],  
    "property_details": ["property_type", "status", "furnish", "construction", "ownership"],
    "specifications": ["facing", "flooring", "builder"],
    "amenities": ["amenities_mcp", "features_mcp", "amenities_text", "overlooking"],
    "location": ["location", "city", "transportation_hubs_clean", "project_name", "nearest_mcp"]
}

def _tokenize_string(text, column_name: str) -> list[str]:
    if pd.isna(text):
        return []
    return [
        word 
        for phrase in clean_and_split(str(text), column_name) 
        for word in str(phrase).split()
    ]

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