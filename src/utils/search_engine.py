# # =====================================================================
# # src/utils/search_engine.py
# # =====================================================================

# from pathlib import Path
# import pandas as pd
# from rank_bm25 import BM25Okapi

# # Cleanly import your production-tested cleaning functions
# from src.utils.search_metadata import clean_and_split

# class RealEstateSearchEngine:
#     def __init__(self, csv_path: Path):
#         print("📥 Loading real estate dataset into search memory...")
#         self.df = pd.read_csv(csv_path).reset_index(drop=True)
        
#         # Cohesive groupings mapped to user search intents
#         self.search_schema = {
#             "bhk": ["bhk_type", "extra_rooms"],
#             "amenities": ["amenities_mcp", "features_mcp", "overlooking"],
#             "location": ["location", "city", "transportation_hubs_clean", "project_name"]
#         }
        
#         self.indexes = {}
#         self._build_bm25_indexes()

#     def _tokenize_row_field(self, text, column_name):
#         """
#         Reuses your exact clean_and_split logic from search_metadata.py
#         to ensure tokens generated for searching match your data structure.
#         """
#         if pd.isna(text):
#             return []
        
#         # clean_and_split returns a list of cleaned terms (e.g. ['cctv surveillance'])
#         cleaned_phrases = clean_and_split(text, column_name)
        
#         # Flatten phrases into loose words/tokens for flexible BM25 matching
#         tokens = []
#         for phrase in cleaned_phrases:
#             tokens.extend(phrase.split())
#         return tokens

#     def _build_bm25_indexes(self):
#         """Pre-computes in-memory search matrices for all categories."""
#         print("🧠 Compiling tokenized BM25 indexes for 11,000+ rows...")
        
#         for category, columns in self.search_schema.items():
#             corpus = []
            
#             # Iterate through the rows to build a combined token profile for this group
#             for _, row in self.df.iterrows():
#                 combined_tokens = []
#                 for col in columns:
#                     if col in self.df.columns:
#                         combined_tokens.extend(self._tokenize_row_field(row[col], col))
#                 corpus.append(combined_tokens)
            
#             # Initialize and store the BM25 index for this specific category group
#             self.indexes[category] = BM25Okapi(corpus)
            
#         print("🚀 BM25 Search matrices successfully compiled!")

#     def query(self, search_criteria: dict, min_matches: int = 2):
#         """
#         Evaluates a parsed user intent dictionary.
#         Filters out rows that don't satisfy the minimum match group threshold.
#         """
#         num_properties = len(self.df)
#         match_counts = [0] * num_properties
#         total_scores = [0.0] * num_properties

#         # Score matching conditions category by category
#         for category, query_str in search_criteria.items():
#             if not query_str or category not in self.indexes:
#                 continue
                
#             # Process query text with the exact same rules as the row fields
#             # We mock 'features_mcp' to pass validation gates nicely
#             query_tokens = []
#             for phrase in clean_and_split(query_str, "features_mcp"):
#                 query_tokens.extend(phrase.split())
                
#             if not query_tokens:
#                 continue
                
#             bm25_index = self.indexes[category]
#             scores = bm25_index.get_scores(query_tokens)
            
#             # Update criteria tracking metrics
#             for idx, score in enumerate(scores):
#                 if score > 0.0:
#                     match_counts[idx] += 1
#                     total_scores[idx] += score

#         # Aggregate properties satisfying the minimum requirement rule
#         matched_records = []
#         for idx in range(num_properties):
#             if match_counts[idx] >= min_matches:
#                 matched_records.append({
#                     "row_idx": idx,
#                     "criteria_matched": match_counts[idx],
#                     "search_score": total_scores[idx]
#                 })
                
#         if not matched_records:
#             return pd.DataFrame()

#         # Build and sort output collection
#         results_df = pd.DataFrame(matched_records).sort_values(
#             by=["criteria_matched", "search_score"], ascending=[False, False]
#         )
        
#         # Merge properties payload back
#         final_df = results_df.merge(self.df, left_on="row_idx", right_index=True)
#         return final_df.drop(columns=["row_idx"])

#===================================================================================================================================================================================================================

# #above code works but some features we have not added above in self.search_schema so that columns we added below 

# # =====================================================================
# # src/utils/search_engine.py
# # =====================================================================

# import pandas as pd
# from rank_bm25 import BM25Okapi

# # Cleanly import your production-tested cleaning functions
# from src.utils.search_metadata import clean_and_split

# class RealEstateSearchEngine:
#     def __init__(self, df: pd.DataFrame):
#         print("📥 Loading real estate dataset into search memory...")
#         self.df = df.reset_index(drop=True)
        
#         # Cohesive groupings mapped to user search intents (All KEEP_COLUMNS accounted for)
#         self.search_schema = {
#             "bhk": ["bhk_type", "extra_rooms"],  
#             "property_details": ["property_type", "status", "furnish", "construction", "ownership"],
#             "specifications": ["facing", "flooring", "builder"],
#             "amenities": ["amenities_mcp", "features_mcp", "amenities_text", "overlooking"],
#             "location": ["location", "city", "transportation_hubs_clean", "project_name", "nearest_mcp"]
#         }
        
#         self.indexes = {}
#         self._build_bm25_indexes()

#     def _tokenize_row_field(self, text, column_name):
#         """Cleans and breaks field text into individual word tokens."""
#         if pd.isna(text):
#             return []
        
#         # clean_and_split returns a list of cleaned terms (e.g. ['cctv surveillance'])
#         cleaned_phrases = clean_and_split(text, column_name)
        
#         # Flatten phrases into loose words/tokens for flexible BM25 matching
#         tokens = []
#         for phrase in cleaned_phrases:
#             tokens.extend(phrase.split())
#         return tokens

#     def _build_bm25_indexes(self):
#         """Pre-computes in-memory search matrices for all categories."""
#         print(f"🧠 Compiling tokenized BM25 indexes for {len(self.df):,} rows...")
        
#         for category, columns in self.search_schema.items():
#             corpus = []
            
#             # Iterate through the rows to build a combined token profile for this group
#             for _, row in self.df.iterrows():
#                 combined_tokens = []
#                 for col in columns:
#                     if col in self.df.columns:
#                         combined_tokens.extend(self._tokenize_row_field(row[col], col))
#                 corpus.append(combined_tokens)
            
#             # Initialize and store the BM25 index for this specific category group
#             self.indexes[category] = BM25Okapi(corpus)
            
#         print("🚀 BM25 Search matrices successfully compiled!")

#     def query(self, search_criteria: dict, min_matches: int = 2):
#         """Scores properties against criteria and filters by minimum category match rules."""
#         num_properties = len(self.df)
#         match_counts = [0] * num_properties
#         total_scores = [0.0] * num_properties

#         # Score matching conditions category by category
#         for category, query_str in search_criteria.items():
#             if not query_str or category not in self.indexes:
#                 continue
                
#             # Process query text with the exact same rules as the row fields
#             # We mock 'features_mcp' to pass validation gates nicely
#             query_tokens = []
#             for phrase in clean_and_split(query_str, "features_mcp"):
#                 query_tokens.extend(phrase.split())
                
#             if not query_tokens:
#                 continue
                
#             bm25_index = self.indexes[category]
#             scores = bm25_index.get_scores(query_tokens)
            
#             # Update criteria tracking metrics
#             for idx, score in enumerate(scores):
#                 if score > 0.0:
#                     match_counts[idx] += 1
#                     total_scores[idx] += score

#         # Aggregate properties satisfying the minimum requirement rule
#         matched_records = []
#         for idx in range(num_properties):
#             if match_counts[idx] >= min_matches:
#                 matched_records.append({
#                     "row_idx": idx,
#                     "criteria_matched": match_counts[idx],
#                     "search_score": total_scores[idx]
#                 })
                
#         if not matched_records:
#             return pd.DataFrame()

#         # Build and sort output collection
#         results_df = pd.DataFrame(matched_records).sort_values(
#             by=["criteria_matched", "search_score"], ascending=[False, False]
#         )
        
#         # Merge properties payload back
#         final_df = results_df.merge(self.df, left_on="row_idx", right_index=True)
#         return final_df.drop(columns=["row_idx"])

#===================================================================================================================================================================================================================

# # =====================================================================
# # src/utils/search_engine.py
# # =====================================================================

# import pandas as pd
# from rank_bm25 import BM25Okapi
# from src.utils.search_metadata import clean_and_split

# # Cohesive groupings mapped to user search intents
# SEARCH_SCHEMA = {
#     "bhk": ["bhk_type", "extra_rooms"],  
#     "property_details": ["property_type", "status", "furnish", "construction", "ownership"],
#     "specifications": ["facing", "flooring", "builder"],
#     "amenities": ["amenities_mcp", "features_mcp", "amenities_text", "overlooking"],
#     "location": ["location", "city", "transportation_hubs_clean", "project_name", "nearest_mcp"]
# }


# def _tokenize_row_field(text, column_name: str) -> list[str]:
#     """Cleans and breaks field text into individual flat word tokens."""
#     if pd.isna(text):
#         return []
    
#     return [
#         word 
#         for phrase in clean_and_split(str(text), column_name) 
#         for word in str(phrase).split()
#     ]


# def _build_bm25_indexes(df: pd.DataFrame) -> dict:
#     """Pre-computes in-memory search matrices and returns the engine state."""
#     print(f"🧠 Compiling tokenized BM25 indexes for {len(df):,} rows...")
    
#     clean_df = df.reset_index(drop=True)
#     indexes = {}

#     for category, columns in SEARCH_SCHEMA.items():
#         valid_cols = [col for col in columns if col in clean_df.columns]
        
#         corpus = [
#             [token for col in valid_cols for token in _tokenize_row_field(row[col], col)]
#             for _, row in clean_df.iterrows()
#         ]
        
#         indexes[category] = BM25Okapi(corpus)
        
#     print("🚀 BM25 Search matrices successfully compiled!")
#     return {"df": clean_df, "indexes": indexes}


# def query(engine_state: dict, search_criteria: dict, min_matches: int = 2) -> pd.DataFrame:
#     """Scores properties against criteria and filters by minimum category match rules."""
#     df = engine_state["df"]
#     indexes = engine_state["indexes"]
    
#     num_properties = len(df)
#     match_counts = [0] * num_properties
#     total_scores = [0.0] * num_properties

#     for category, query_str in search_criteria.items():
#         if not query_str or category not in indexes:
#             continue
            
#         query_tokens = _tokenize_row_field(query_str, "features_mcp")
#         if not query_tokens:
#             continue
            
#         scores = indexes[category].get_scores(query_tokens)
        
#         for idx, score in enumerate(scores):
#             if score > 0.0:
#                 match_counts[idx] += 1
#                 total_scores[idx] += score

#     matched_records = [
#         {"row_idx": idx, "criteria_matched": match_counts[idx], "search_score": total_scores[idx]}
#         for idx in range(num_properties)
#         if match_counts[idx] >= min_matches
#     ]
        
#     if not matched_records:
#         return pd.DataFrame()

#     results_df = pd.DataFrame(matched_records).sort_values(
#         by=["criteria_matched", "search_score"], 
#         ascending=[False, False]
#     )
    
#     final_df = results_df.merge(df, left_on="row_idx", right_index=True)
#     return final_df.drop(columns=["row_idx"])

#============================================================================================================================================================================

# ===================================================================== 
# REFACTORED: src/utils/search_engine.py (PURE EXECUTION ENGINE)
# =====================================================================

import pandas as pd
from src.utils.search_metadata import clean_and_split

def _tokenize_row_field(text, column_name: str) -> list[str]:
    """Cleans and breaks field text into individual flat word tokens."""
    if pd.isna(text):
        return []
    
    return [
        word 
        for phrase in clean_and_split(str(text), column_name) 
        for word in str(phrase).split()
    ]

def query(engine_state: dict, search_criteria: dict, min_matches: int = 2) -> pd.DataFrame:
    """Scores properties against criteria using a pre-computed external engine state."""
    df = engine_state["df"]
    indexes = engine_state["indexes"]
    
    num_properties = len(df)
    match_counts = [0] * num_properties
    total_scores = [0.0] * num_properties

    for category, query_str in search_criteria.items():
        if not query_str or category not in indexes:
            continue
            
        query_tokens = _tokenize_row_field(query_str, "features_mcp")
        if not query_tokens:
            continue
            
        scores = indexes[category].get_scores(query_tokens)
        
        for idx, score in enumerate(scores):
            if score > 0.0:
                match_counts[idx] += 1
                total_scores[idx] += score

    matched_records = [
        {"row_idx": idx, "criteria_matched": match_counts[idx], "search_score": total_scores[idx]}
        for idx in range(num_properties)
        if match_counts[idx] >= min_matches
    ]
        
    if not matched_records:
        return pd.DataFrame()

    results_df = pd.DataFrame(matched_records).sort_values(
        by=["criteria_matched", "search_score"], 
        ascending=[False, False]
    )
    
    final_df = results_df.merge(df, left_on="row_idx", right_index=True)
    return final_df.drop(columns=["row_idx"])