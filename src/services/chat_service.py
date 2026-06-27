# # ===============================
# # chat_service.py
# # ===============================

# def build_context(
#     recs=None,
#     comparison_result=None,
#     comparison_raw=None,
#     last_explanation=None
# ):
#     """
#     Build compact property context for LLM.

#     Priority:
#     1. Input property
#     2. Comparison raw (full enriched property data)
#     3. Comparison result (summary scores/verdict)
#     4. Explanation
#     """

#     sections = []

#     # =====================================
#     # INPUT PROPERTY
#     # =====================================
#     # Extract selected columns and their values from recs["input"] and add them to the sections list for LLM context generation.
#     if recs and "input" in recs:

#         input_df = recs["input"]

#         if not input_df.empty:

#             cols = [
#                 c for c in [
#                     "id",
#                     "project_name",
#                     "location",
#                     "price",
#                     "area",
#                     "bhk_type"
#                 ]
#                 if c in input_df.columns
#             ]

#             sections.append(
#                 "INPUT PROPERTY:\n"
#                 + input_df[cols].to_string(index=False)
#             )

#     # =====================================
#     # ENRICHED PROPERTY DATA
#     # =====================================
#     if (
#         comparison_raw is not None
#         and not comparison_raw.empty
#     ):

#         important_cols = [

#             # identity
#             "id",
#             "project_name",
#             "location",

#             # recommendation
#             "why_recommended",
#             "hybrid_score",

#             # valuation
#             "analysis_msg",

#             # risk
#             "risk_label",
#             "risk_score",

#             # growth
#             "growth_label",
#             "growth_reason",

#             # rental
#             "monthly_rent_estimate",
#             "rental_yield_percent",
#             "investment_rating",

#             # negotiation
#             "negotiation_power",
#             "suggested_discount_percent",
#             "target_price",
#             "price_position",

#             # development
#             "dev_summary"
#         ]

#         cols = [
#             c
#             for c in important_cols
#             if c in comparison_raw.columns
#         ]

#         sections.append(
#             "PROPERTY DATA:\n"
#             + comparison_raw[cols].to_string(index=False)
#         )

#     # =====================================
#     # COMPARISON SUMMARY
#     # =====================================
#     if (
#         comparison_result is not None
#         and not comparison_result.empty
#     ):

#         cols = [
#             c for c in [
#                 "id",
#                 "overall_score",
#                 "verdict",
#                 "comparison_reason"
#             ]
#             if c in comparison_result.columns
#         ]

#         sections.append(
#             "COMPARISON SUMMARY:\n"
#             + comparison_result[cols].to_string(index=False)
#         )

#     # =====================================
#     # EXPLANATION
#     # =====================================
#     if last_explanation:

#         sections.append(
#             "COMPARISON INSIGHTS:\n"
#             + str(last_explanation)
#         )

#     if not sections:
#         return "No property data available."

#     return "\n\n".join(sections)

#===========================================================================================

# # so this new below code is made for the user intent project and we have keep this above earlier code
# # as it is below 100% just added new extra function parse_intent_and_execute


# # ===============================
# # chat_service.py
# # ===============================
# import re
# import json
# import pandas as pd
# from src.utils.search_engine import _build_bm25_indexes, query
# from src.mcp.tools.comparison_tools import compare_properties
# from src.llm.memory_store import SQLiteMemoryStore
# from src.mcp.tools.rental_tools import (
#     get_rental_analysis
# )
# import streamlit as st
# from src.data.data_store import master_df

# search_state = _build_bm25_indexes(master_df)
# memory_store = SQLiteMemoryStore()

# USER_ID = "default_user"

# # Operational tracking filters
# CHAT_STOPWORDS = {
#     "want", "need", "show", "me", "find", "get", "property", "properties",
#     "with", "and", "a", "an", "the", "for", "in", "at", "house", "flat", 
#     "apartment", "please", "give", "share", "only", "near", "to", "from", "but", "i"
# }

# def build_context(
#     recs=None,
#     comparison_result=None,
#     comparison_raw=None,
#     last_explanation=None
# ):
#     """
#     Build compact property context for LLM.

#     Priority:
#     1. Input property
#     2. Comparison raw (full enriched property data)
#     3. Comparison result (summary scores/verdict)
#     4. Explanation
#     """

#     sections = []

#     # =====================================
#     # INPUT PROPERTY
#     # =====================================
#     # Extract selected columns and their values from recs["input"] and add them to the sections list for LLM context generation.
#     if recs and "input" in recs:

#         input_df = recs["input"]

#         if not input_df.empty:

#             cols = [
#                 c for c in [
#                     "id",
#                     "project_name",
#                     "location",
#                     "price",
#                     "area",
#                     "bhk_type"
#                 ]
#                 if c in input_df.columns
#             ]

#             sections.append(
#                 "INPUT PROPERTY:\n"
#                 + input_df[cols].to_string(index=False)
#             )

#     # =====================================
#     # ENRICHED PROPERTY DATA
#     # =====================================
#     if (
#         comparison_raw is not None
#         and not comparison_raw.empty
#     ):

#         important_cols = [

#             # identity
#             "id",
#             "project_name",
#             "location",

#             # recommendation
#             "why_recommended",
#             "hybrid_score",

#             # valuation
#             "analysis_msg",

#             # risk
#             "risk_label",
#             "risk_score",

#             # growth
#             "growth_label",
#             "growth_reason",

#             # rental
#             "monthly_rent_estimate",
#             "rental_yield_percent",
#             "investment_rating",

#             # negotiation
#             "negotiation_power",
#             "suggested_discount_percent",
#             "target_price",
#             "price_position",

#             # development
#             "dev_summary"
#         ]

#         cols = [
#             c
#             for c in important_cols
#             if c in comparison_raw.columns
#         ]

#         sections.append(
#             "PROPERTY DATA:\n"
#             + comparison_raw[cols].to_string(index=False)
#         )

#     # =====================================
#     # COMPARISON SUMMARY
#     # =====================================
#     if (
#         comparison_result is not None
#         and not comparison_result.empty
#     ):

#         cols = [
#             c for c in [
#                 "id",
#                 "overall_score",
#                 "verdict",
#                 "comparison_reason"
#             ]
#             if c in comparison_result.columns
#         ]

#         sections.append(
#             "COMPARISON SUMMARY:\n"
#             + comparison_result[cols].to_string(index=False)
#         )

#     # =====================================
#     # EXPLANATION
#     # =====================================
#     if last_explanation:

#         sections.append(
#             "COMPARISON INSIGHTS:\n"
#             + str(last_explanation)
#         )

#     if not sections:
#         return "No property data available."

#     return "\n\n".join(sections)


# def extract_search_tokens(text: str):
#     """Isolates pristine structural search tokens from human entries."""
#     raw_words = text.lower().strip().split()
#     # Heavily check and drop anything containing 'bhk' or matching noise tokens
#     clean_tokens = [w for w in raw_words if w not in CHAT_STOPWORDS and "bhk" not in w]
#     return " ".join(clean_tokens) if clean_tokens else None


# def parse_intent_and_execute(user_prompt: str, session_state_tray: list) -> dict:
#     """Parses user requests to run property comparisons or rental analysis."""

#     prompt_lower = user_prompt.lower().strip()
#     words = prompt_lower.split()
    
#     # -----------------------------------------------------------------
#     # STEP 1: Comparison Analysis Trigger
#     # -----------------------------------------------------------------
#     comparison_keywords = [
#         "compare",
#         "ranking",
#         "rank"
#     ]

#     if any(k in prompt_lower for k in comparison_keywords):
#         if len(session_state_tray) < 2:
#             return {
#                 "type": "text",
#                 "content": "⚠️ I need at least 2 properties in your tray to run an investment comparison. Try searching and adding properties first!"
#             }
        
#         print("🔄 Routing context directly to Comparison Node Module...")
#         comparison_raw_json = compare_properties(session_state_tray)
#         comparison_data = json.loads(comparison_raw_json)
        
#         return {
#             "type": "comparison",
#             "content": comparison_data
#         }
    
#     rental_keywords = [
#         "rent",
#         "rental",
#         "yield",
#         "income"
#     ]

#     if any(
#         k in prompt_lower
#         for k in rental_keywords
#     ):

#         if len(session_state_tray) < 1:
#             return {
#                 "type": "text",
#                 "content":
#                 "Please add property to tray first."
#             }

#         rental_json = get_rental_analysis(
#             session_state_tray
#         )

#         return {
#             "type": "rental",
#             "content": json.loads(
#                 rental_json
#             )
#         }

#     # -----------------------------------------------------------------
#     # STEP 2: Structural Parameter Isolation
#     # -----------------------------------------------------------------
#     extracted_criteria = {"bhk": None, "amenities": None, "location": None}
    
#     # Isolate BHK constraints cleanly
#     match = re.search(
#         r'(\d+)\s*bhk',
#         prompt_lower
#     )

#     if match:
#         extracted_criteria["bhk"] = f"{match.group(1)}bhk"
#     else:
#         for idx, word in enumerate(words):
#             if word == "bhk" and idx > 0 and words[idx-1].isdigit():
#                 extracted_criteria["bhk"] = f"{words[idx-1]}bhk"
#                 break

#     # Positional separator parsing (Checks for 'near' or 'from')
#     separator_word = None
#     if "near" in words:
#         separator_word = "near"
#     elif "from" in words:
#         separator_word = "from"

#     if separator_word:
#         sep_idx = words.index(separator_word)
#         extracted_criteria["location"] = extract_search_tokens(" ".join(words[sep_idx + 1:]))
#         extracted_criteria["amenities"] = extract_search_tokens(" ".join(words[:sep_idx]))
#     else:
#         # If no strict separator, clean the whole string for attributes
#         extracted_criteria["amenities"] = extract_search_tokens(user_prompt)
#         extracted_criteria["location"] = extract_search_tokens(user_prompt)

#     # -----------------------------------------------------------------
#     # STEP 3: Persistent SQLite Long-Term Memory Lookup & Merge
#     # -----------------------------------------------------------------
#     saved_memories = memory_store.get_memories(USER_ID)
#     historical_state = {}
#     for memory_string in saved_memories:
#         try:
#             parsed_block = json.loads(memory_string)
#             if isinstance(parsed_block, dict):
#                 historical_state.update(parsed_block)
#         except json.JSONDecodeError:
#             continue

#     # --- THE CRITICAL BUG FIX LAYER ---
#     # Only fall back to old memory if the user did NOT type a fresh location in their current prompt!
#     if not extracted_criteria["location"]:
#         # If no fresh location token was provided, pass forward old location memory safely
#         if historical_state.get("location"):
#             extracted_criteria["location"] = historical_state["location"]
#     else:
#         pass

#     if not extracted_criteria["bhk"] and historical_state.get("bhk"):
#         extracted_criteria["bhk"] = historical_state["bhk"]

#     # -----------------------------------------------------------------
#     # STEP 4: Persist the Fresh Combined Query back to SQLite
#     # -----------------------------------------------------------------
#     active_preferences = {k: v for k, v in extracted_criteria.items() if v}
#     if active_preferences:
#         memory_store.add_memory(USER_ID, json.dumps(active_preferences))

#     print(f"🧠 SQLite Merged System State Processing: {extracted_criteria}")
    
#     # -----------------------------------------------------------------
#     # STEP 5: Fire BM25 Matrix Matcher + Pandas HARD FILTERING
#     # -----------------------------------------------------------------
#     determined_min_matches = 1 if (not extracted_criteria["amenities"] or not extracted_criteria["location"]) else 2
#     results_df = query(search_state, extracted_criteria, min_matches=determined_min_matches)
    
#     if results_df.empty:
#         return {
#             "type": "text",
#             "content": f"❌ Zero properties matched your search parameters: `{extracted_criteria}`."
#         }
    
#     # --- THE CRITICAL HARD FILTERING LAYER ---
#     # If a specific BHK value exists in our search state, forcefully prune the output frame
#     if extracted_criteria["bhk"]:
#         target_bhk = extracted_criteria["bhk"].strip().lower() # e.g., "1 bhk" or "2 bhk"
        
#         # Build an exact string matching mask against your df['bhk_type'] column
#         if "bhk_type" in results_df.columns:
#             # We standardize comparisons to handle potential spacing variations (e.g., "2bhk" vs "2 bhk")
#             filtered_df = results_df[
#                 results_df["bhk_type"].str.lower().str.replace(" ", "") == target_bhk.replace(" ", "")
#             ]
            
#             # Fallback: If strict filtering leaves us with a completely empty frame, 
#             # gracefully return original BM25 results but print a terminal notice.
#             if not filtered_df.empty:
#                 results_df = filtered_df
#             else:
#                 print(f"⚠️ Warning: Strict filter for '{target_bhk}' returned zero rows. Falling back to fuzzy matches.")

#     return {
#         "type": "search_results",
#         "content": results_df.head(5).to_dict(orient="records"),
#         "current_query_state": extracted_criteria
#     }


#==========================================================================================


# this below code is exact above just added prediction,advisor,negotiation,valuation tools


# # ===============================
# # chat_service.py
# # ===============================
# import re
# import json
# import pandas as pd
# import streamlit as st

# from src.utils.search_engine import _build_bm25_indexes, query
# from src.llm.memory_store import SQLiteMemoryStore
# from src.data.data_store import master_df
# from src.llm.deepseek_client import ask_deepseek

# # Unified MCP Tools Layer
# from src.mcp.tools.property_tools import (
#     compare_properties,
#     get_rental_analysis,
#     get_price_prediction,
#     get_negotiation_strategy,
#     get_valuation_analysis,
#     get_investment_advice
# )

# # Core initializations
# search_state = _build_bm25_indexes(master_df)
# memory_store = SQLiteMemoryStore()

# USER_ID = "default_user"

# # Operational tracking filters
# CHAT_STOPWORDS = {
#     "want", "need", "show", "me", "find", "get", "property", "properties",
#     "with", "and", "a", "an", "the", "for", "in", "at", "house", "flat", 
#     "apartment", "please", "give", "share", "only", "near", "to", "from", "but", "i"
# }

# def build_context(
#     recs=None,
#     comparison_result=None,
#     comparison_raw=None,
#     last_explanation=None
# ):
#     """
#     Build compact property context for LLM.

#     Priority:
#     1. Input property
#     2. Comparison raw (full enriched property data)
#     3. Comparison result (summary scores/verdict)
#     4. Explanation
#     """

#     sections = []

#     # ====================================
#     # INPUT PROPERTY
#     # =====================================
#     # Extract selected columns and their values from recs["input"] and add them to the sections list for LLM context generation.
#     if recs and "input" in recs:
#         input_df = recs["input"]

#         if not input_df.empty:

#             cols = [
#                 c for c in [
#                     "id",
#                     "project_name",
#                     "location",
#                     "price",
#                     "area",
#                     "bhk_type"
#                 ]
#                 if c in input_df.columns
#             ]

#             sections.append(
#                 "INPUT PROPERTY:\n"
#                 + input_df[cols].to_string(index=False)
#             )

#     # =====================================
#     # ENRICHED PROPERTY DATA
#     # =====================================
#     if comparison_raw is not None and not comparison_raw.empty:
#         important_cols = [
#             "id", "project_name", "location",
#             "why_recommended", "hybrid_score",
#             "analysis_msg",
#             "risk_label", "risk_score",
#             "growth_label", "growth_reason",
#             "monthly_rent_estimate", "rental_yield_percent", "investment_rating",
#             "negotiation_power", "suggested_discount_percent", "target_price", "price_position",
#             "dev_summary"
#         ]
#         cols = [c for c in important_cols if c in comparison_raw.columns]
#         sections.append(
#             "PROPERTY DATA:\n"
#             + comparison_raw[cols].to_string(index=False)
#         )

#     # =====================================
#     # COMPARISON SUMMARY
#     # =====================================
#     if comparison_result is not None and not comparison_result.empty:
#         cols = [
#             c for c in [
#                 "id",
#                 "overall_score",
#                 "verdict",
#                 "comparison_reason"
#             ]
#             if c in comparison_result.columns
#         ]
#         sections.append(
#             "COMPARISON SUMMARY:\n"
#             + comparison_result[cols].to_string(index=False)
#         )

#     # =====================================
#     # EXPLANATION
#     # =====================================
#     if last_explanation:

#         sections.append(
#             "COMPARISON INSIGHTS:\n"
#             + str(last_explanation)
#         )

#     if not sections:
#         return "No property data available."

#     return "\n\n".join(sections)


# def extract_search_tokens(text: str):
#     """Isolates pristine structural search tokens from human entries."""
#     raw_words = text.lower().strip().split()
#     # Heavily check and drop anything containing 'bhk' or matching noise tokens
#     clean_tokens = [w for w in raw_words if w not in CHAT_STOPWORDS and "bhk" not in w]
#     return " ".join(clean_tokens) if clean_tokens else None


# def parse_intent_and_execute(user_prompt: str, session_state_tray: list) -> dict:
#     """Evaluates multi-node execution signals, fallback states, or LLM queries."""

#     prompt_lower = user_prompt.lower().strip()
#     words = prompt_lower.split()
    
#     # -----------------------------------------------------------------
#     # STEP 1: EVALUATION CORE MATRIX ROUTING
#     # -----------------------------------------------------------------
    
#     # ROUTE A: COMPARISON ENGINE
#     if any(k in prompt_lower for k in ["compare", "ranking", "rank"]):
#         if len(session_state_tray) < 2:
#             return {
#                 "type": "text", 
#                 "content": "⚠️ I need at least 2 properties in your tray to run an investment comparison. Try searching and adding properties first!"
#             }
#         return {"type": "comparison", "content": json.loads(compare_properties(session_state_tray))}
        
#     # ROUTE B: RENTAL MATRIX
#     if any(k in prompt_lower for k in ["rent", "rental", "yield", "income"]):
#         if len(session_state_tray) < 1:
#             return {
#                 "type": "text", 
#                 "content": "⚠️ Please add at least one target property to your evaluation tray first."
#             }
#         return {"type": "rental", "content": json.loads(get_rental_analysis(session_state_tray))}

#     # ROUTE C: MODEL PRICE PREDICTION
#     if any(k in prompt_lower for k in ["predict", "prediction", "forecast", "estimated price"]):
#         if len(session_state_tray) < 1:
#             return {
#                 "type": "text", 
#                 "content": "⚠️ Your evaluation tray is empty. Stage properties from search results to run price predictions."
#             }
#         return {"type": "prediction", "content": json.loads(get_price_prediction(session_state_tray))}

#     # ROUTE D: NEGOTIATION ADVICE
#     if any(k in prompt_lower for k in ["negotiate", "negotiation", "discount", "leverage", "target price"]):
#         if len(session_state_tray) < 1:
#             return {
#                 "type": "text", 
#                 "content": "⚠️ No context found in evaluation tray. Stage items first to map strategic talking points."
#             }
#         return {"type": "negotiation", "content": json.loads(get_negotiation_strategy(session_state_tray))}

#     # ROUTE E: VALUATION ANALYTICS
#     if any(k in prompt_lower for k in ["overpriced", "undervalued", "fair value", "valuation"]):
#         if len(session_state_tray) < 1:
#             return {
#                 "type": "text", 
#                 "content": "⚠️ Insufficient context. Stage properties to calculate fair-market valuation thresholds."
#             }
#         return {"type": "valuation", "content": json.loads(get_valuation_analysis(session_state_tray))}

#     # ROUTE F: INVESTMENT ADVISOR
#     if any(k in prompt_lower for k in ["should i buy", "investment advice", "advisor", "suitability", "positives"]):
#         if len(session_state_tray) < 1:
#             return {
#                 "type": "text", 
#                 "content": "⚠️ Please stage matching properties in your comparison tray to extract advisor insights."
#             }
#         return {"type": "advisor", "content": json.loads(get_investment_advice(session_state_tray))}

#     # -----------------------------------------------------------------
#     # STEP 2: Structural Parameter Isolation
#     # -----------------------------------------------------------------
#     extracted_criteria = {"bhk": None, "amenities": None, "location": None}
    
#     # Isolate BHK constraints cleanly
#     match = re.search(r'(\d+)\s*bhk', prompt_lower)
#     if match:
#         extracted_criteria["bhk"] = f"{match.group(1)}bhk"
#     else:
#         for idx, word in enumerate(words):
#             if word == "bhk" and idx > 0 and words[idx-1].isdigit():
#                 extracted_criteria["bhk"] = f"{words[idx-1]}bhk"
#                 break

#     # Positional separator parsing (Checks for 'near' or 'from')
#     separator_word = None
#     if "near" in words:
#         separator_word = "near"
#     elif "from" in words:
#         separator_word = "from"

#     if separator_word:
#         sep_idx = words.index(separator_word)
#         extracted_criteria["location"] = extract_search_tokens(" ".join(words[sep_idx + 1:]))
#         extracted_criteria["amenities"] = extract_search_tokens(" ".join(words[:sep_idx]))
#     else:
#         # If no strict separator, clean the whole string for attributes
#         extracted_criteria["amenities"] = extract_search_tokens(user_prompt)
#         extracted_criteria["location"] = extract_search_tokens(user_prompt)

#     # -----------------------------------------------------------------
#     # STEP 3: Persistent SQLite Long-Term Memory Lookup & Merge
#     # -----------------------------------------------------------------
#     saved_memories = memory_store.get_memories(USER_ID)
#     historical_state = {}
#     for memory_string in saved_memories:
#         try:
#             parsed_block = json.loads(memory_string)
#             if isinstance(parsed_block, dict):
#                 historical_state.update(parsed_block)
#         except json.JSONDecodeError:
#             continue

#     # --- THE CRITICAL BUG FIX LAYER ---
#     # Only fall back to old memory if the user did NOT type a fresh location in their current prompt!
#     if not extracted_criteria["location"]:
#         # If no fresh location token was provided, pass forward old location memory safely
#         if historical_state.get("location"):
#             extracted_criteria["location"] = historical_state["location"]

#     if not extracted_criteria["bhk"] and historical_state.get("bhk"):
#         extracted_criteria["bhk"] = historical_state["bhk"]

#     # -----------------------------------------------------------------
#     # STEP 4: Persist the Fresh Combined Query back to SQLite
#     # -----------------------------------------------------------------
#     active_preferences = {k: v for k, v in extracted_criteria.items() if v}
#     if active_preferences:
#         memory_store.add_memory(USER_ID, json.dumps(active_preferences))

#     print(f"🧠 SQLite Merged System State Processing: {extracted_criteria}")
    
#     # -----------------------------------------------------------------
#     # STEP 5: Fire BM25 Matrix Matcher + Pandas HARD FILTERING
#     # -----------------------------------------------------------------
#     if extracted_criteria["location"] or extracted_criteria["amenities"] or extracted_criteria["bhk"]:
#         determined_min_matches = 1 if (not extracted_criteria["amenities"] or not extracted_criteria["location"]) else 2
#         results_df = query(search_state, extracted_criteria, min_matches=determined_min_matches)
    
#         if not results_df.empty:
#                 # --- THE CRITICAL HARD FILTERING LAYER ---
        
#             # --- THE CRITICAL HARD FILTERING LAYER ---
#             # If a specific BHK value exists in our search state, forcefully prune the output frame
#             if extracted_criteria["bhk"]:
#                 target_bhk = extracted_criteria["bhk"].strip().lower() # e.g., "1 bhk" or "2 bhk"
                
#                 # Build an exact string matching mask against your df['bhk_type'] column
#                 if "bhk_type" in results_df.columns:
#                     # We standardize comparisons to handle potential spacing variations (e.g., "2bhk" vs "2 bhk")
#                     filtered_df = results_df[
#                         results_df["bhk_type"].str.lower().str.replace(" ", "") == target_bhk.replace(" ", "")
#                     ]
                    
#                     # Fallback: If strict filtering leaves us with a completely empty frame, 
#                     # gracefully return original BM25 results but print a terminal notice.
#                     if not filtered_df.empty:
#                         results_df = filtered_df
#                     else:
#                         print(f"⚠️ Warning: Strict filter for '{target_bhk}' returned zero rows. Falling back to fuzzy matches.")

#             return {
#                 "type": "search_results",
#                 "content": results_df.head(5).to_dict(orient="records"),
#                 "current_query_state": extracted_criteria
#             }
#         else:
#             return {
#                 "type": "text",
#                 "content": f"❌ Zero properties matched your search parameters: `{extracted_criteria}`."
#             }

#     # -----------------------------------------------------------------
#     # ROUTE G: GENERAL CHAT FALLBACK LAYER
#     # -----------------------------------------------------------------
#     staged_context = master_df[master_df["id"].isin(session_state_tray)].head(3).to_string(index=False) if session_state_tray else "No active properties staged."
    
#     chat_prompt = f"""You are an expert real estate consultant. Answer the user's inquiry directly.
    
#     ACTIVE STAGED PROPERTIES IN USER'S TRAY CONTEXT:
#     {staged_context}
    
#     USER QUERY: {user_prompt}
    
#     Provide a practical, clear response using Indian Rupee (₹) formats for all real estate evaluations. Keep your response concise.
#     """
    
#     response_text = ask_deepseek(chat_prompt)
#     return {
#         "type": "text",
#         "content": response_text
#     }

#=======================================================================================================================================================================
# # ================================
# # chat_service.py
# # ================================
# """
# Refactored Streamlit Chat Processing Engine.
# Cleaned up faulty variable exports to ensure high-speed runtime execution.
# """
# import re
# import json
# import pandas as pd
# import streamlit as st

# import src.mcp.tools.property_tools as tools
# # ADDED: Explicitly import CACHED_SEARCH_METADATA to handle keyword sweeps
# from src.core.search_registry import GLOBAL_MASTER_DF, CACHED_SEARCH_METADATA
# from src.llm.memory_store import SQLiteMemoryStore
# from src.llm.deepseek_client import ask_deepseek

# memory_store = SQLiteMemoryStore()
# USER_ID = "default_user"

# def parse_intent_and_execute(user_prompt: str, session_state_tray: list) -> dict:
#     prompt_lower = user_prompt.lower().strip()
#     # This function reads the user's question, figures out what analysis they want (comparison, rental, prediction, negotiation, valuation, or investment advice), 
#     # checks if enough properties are available in the tray, and then calls the appropriate tool to generate the result.

#     if any(k in prompt_lower for k in ["compare", "ranking", "rank"]):
#         if len(session_state_tray) < 2:
#             return {"type": "text", "content": "⚠️ I need at least 2 properties in your tray to run an investment comparison."}
#         return {"type": "comparison", "content": tools.compare_properties(session_state_tray)}
        
#     if any(k in prompt_lower for k in ["rent", "rental", "tenant", "lease", "yield", "rental yield", "monthly rent", "annual rent", "rental estimate", "rental income", "income property", "yield", "income"]): 
#         if len(session_state_tray) < 1:
#             return {"type": "text", "content": "⚠️ Please add at least one target property to your evaluation tray first."}
#         return {"type": "rental", "content": tools.get_rental_analysis(session_state_tray)}

#     if any(k in prompt_lower for k in ["predict", "prediction", "predicted price", "estimated price", "price estimate", "price prediction", "property value", "future price", "what should this cost", "forecast"]):
#         if len(session_state_tray) < 1:
#             return {"type": "text", "content": "⚠️ Your evaluation tray is empty. Stage properties to run predictions."}
#         return {"type": "prediction", "content": tools.get_price_prediction(session_state_tray)}

#     if any(k in prompt_lower for k in ["negotiate", "negotiation", "negotiable", "discount", "best price", "reduce price", "target price", "deal", "bargain", "leverage"]):
#         if len(session_state_tray) < 1:
#             return {"type": "text", "content": "⚠️ No context found in evaluation tray. Stage items first."}
#         return {"type": "negotiation", "content": tools.get_negotiation_strategy(session_state_tray)}

#     if any(k in prompt_lower for k in ["overpriced", "undervalued", "valuation", "fair value", "fair price", "worth buying", "worth it", "market value"]):
#         if len(session_state_tray) < 1:
#             return {"type": "text", "content": "⚠️ Insufficient context. Stage properties to calculate valuation parameters."}
#         return {"type": "valuation", "content": tools.get_valuation_analysis(session_state_tray)}

#     if any(k in prompt_lower for k in ["should i buy", "buy this property", "is this a good investment", "investment advice", "recommendation", "final advice", "which property should i buy", "best investment", "best property", "advisor", "why did you mark", "why buy", "show exact scores", "show scores", "advisor score", "buy decision", "investment decision", "recommendation reason", "why recommendation", "suitability", "positives"]):
#         if len(session_state_tray) < 1:
#             return {"type": "text", "content": "⚠️ Please stage matching properties in your comparison tray first."}
#         return {"type": "advisor", "content": tools.get_investment_advice(session_state_tray)}

#     # -----------------------------------------------------------------
#     # STEP 2: DYNAMIC CONTEXT TOKEN GENERATION (FIXED & BHK HARD SYNCED)
#     # -----------------------------------------------------------------
#     extracted_criteria = {"bhk": None, "amenities": None, "location": None}
    
#     # 1. Regex Extraction for BHK (e.g., "2bhk", "3 bhk")
#     # FIX: Append "bhk" directly to the isolated digit to pass an exact criteria match string
#     match = re.search(r'(\d+)\s*bhk', prompt_lower)
#     if match:
#         extracted_criteria["bhk"] = f"{match.group(1)}bhk"
        
#     # 2. Local Keyword Extraction via Pristine Metadata Cache
#     matched_locations = []
#     matched_amenities = []
    
#     known_locations = CACHED_SEARCH_METADATA.get("location", [])
#     known_amenities = CACHED_SEARCH_METADATA.get("amenities_mcp", [])
    
#     # Scan for known locations in the prompt
#     for loc in known_locations:
#         if str(loc).lower() in prompt_lower:
#             matched_locations.append(loc)
            
#     # Scan for known amenities/features in the prompt
#     for amenity in known_amenities:
#         if str(amenity).lower() in prompt_lower:
#             matched_amenities.append(amenity)
            
#     # Join matches together cleanly into flat search criteria strings for the BM25 query engine
#     if matched_locations:
#         extracted_criteria["location"] = " ".join(matched_locations)
#     if matched_amenities:
#         extracted_criteria["amenities"] = " ".join(matched_amenities)

#     # -----------------------------------------------------------------
#     # STEP 3: HIGH-SPEED LOCAL SEARCH HANDLING
#     # -----------------------------------------------------------------
#     if extracted_criteria["location"] or extracted_criteria["amenities"] or extracted_criteria["bhk"]:
#         results = tools.search_properties(
#             bhk=extracted_criteria["bhk"], 
#             amenities=extracted_criteria["amenities"], 
#             location=extracted_criteria["location"]
#         )
#         if results:
#             return {
#                 "type": "search_results",
#                 "content": results,
#                 "current_query_state": extracted_criteria
#             }
#         else:
#             return {"type": "text", "content": f"❌ Zero properties matched your search parameters: `{extracted_criteria}`."}

#     # -----------------------------------------------------------------
#     # STEP 4: SEMANTIC DEEPSEEK AGENT FALLBACK LAYER
#     # -----------------------------------------------------------------
#     staged_context = GLOBAL_MASTER_DF[GLOBAL_MASTER_DF["id"].isin(session_state_tray)].head(3).to_string(index=False) if session_state_tray else "No active properties staged."
#     chat_prompt = f"""You are an expert real estate consultant. Answer the inquiry directly.
    
#     ACTIVE CONTEXT ROWS IN USER MEMORY TRAY:
#     {staged_context}
    
#     USER REQUEST INPUTS: {user_prompt}
#     Provide structured clear insights utilizing Indian Rupee (₹) denominations.
#     """
#     return {"type": "text", "content": ask_deepseek(chat_prompt)}


#==========================================================================================================================================================================================

# # =====================================================================
# # chat_service.py (PRODUCTION ARCHITECTURE - FIXED MATCH QUALITY)
# # =====================================================================

# import re
# import json
# import pandas as pd
# import streamlit as st

# import src.mcp.tools.property_tools as tools
# from src.core.search_registry import GLOBAL_MASTER_DF, CACHED_SEARCH_METADATA
# from src.llm.memory_store import SQLiteMemoryStore
# from src.llm.deepseek_client import ask_deepseek
# from src.recommender.hybrid_recommender import apply_hybrid_ranking

# memory_store = SQLiteMemoryStore()
# USER_ID = "default_user"

# # =====================================================================
# # CONFIGURATION MATRICES
# # =====================================================================

# # Structural hard constraints
# FILTER_INTENTS = {
#     "bhk_pattern":      r"(\d+)\s*bhk",
#     "known_locations":  "location",      
#     "known_amenities":  "amenities_mcp"  
# }

# # Cleaned naming mapping dictionary to uppercase standard
# RANKING_TARGET_MAPS = {
#     "low budget":      {"price": 1.0},
#     "luxury":          {"amenities": 1.0, "area": 0.8, "location": 0.4}, 
#     "spacious":        {"area": 1.0},
#     "good amenities":  {"amenities": 1.0},
#     "connectivity":    {"connectivity": 1.0, "distance": 0.6},
#     "location":        {"location": 1.0},
#     "investment":      {"price": 1.0, "location": 0.5},
#     "family":          {"location": 1.0, "area": 0.5}
# }

# RANKING_WORD_LISTS = {
#     "low budget":      ["low budget", "cheap", "affordable", "under budget", "value for money", "pocket friendly", "lowest price", "pocket-friendly", "economical"],
#     "luxury":          ["luxury", "premium", "posh", "high end", "elite", "luxurious", "expensive", "high-end", "ultra-premium"],
#     "spacious":        ["spacious", "big size", "large area", "huge", "roomy", "bigger rooms", "carpet area", "massive square feet"],
#     "good amenities": ["good amenities", "premium community features", "high-end facilities"],
#     "connectivity":    ["near station", "metro", "railway", "connectivity", "public transport", "highway", "link road", "walkable", "easy commuting", "commuting is easier", "access to office", "less travel time", "save travel time"],
#     "location":        ["great location", "prime location", "well located", "center of city", "heart of mumbai"],
#     "investment":      ["investment", "good yield", "resale value", "future growth", "high return", "roi", "appreciation"],
#     "family":          ["family oriented", "safe neighborhood", "school proximity", "gated community"]
# }

# # Advanced Modifier Token Layers
# NEGATIONS = [r"not\b", r"don't\b", r"do\s+not\b", r"without\b", r"avoid\b", r"no\b", r"never\b"]

# INTENSITY_MODIFIERS = {
#     "extremely": 2.0, "super": 1.5, "very": 1.4, "highly": 1.4,
#     "good": 1.1, "great": 1.2, "prime": 1.2, "ultra": 2.0
# }

# # Followup Continuity Tracking Layers
# FOLLOWUP_TERMS = [
#     r"similar\b", r"show\s+more\b", r"cheaper\b", r"better\b", 
#     r"closer\b", r"other\s+options\b", r"nearer\b", r"also\b"
# ]

# # =====================================================================
# # SAFE MEMORY INTERACTION LAYER
# # =====================================================================

# def fetch_safe_historical_context(store_instance, user_id: str) -> dict:
#     """
#     Safely bridges SQLiteMemoryStore hook differences 
#     by testing available persistence method hooks.
#     """
#     for method_name in ["get_latest_context", "load", "get_context", "get_history"]:
#         if hasattr(store_instance, method_name):
#             method = getattr(store_instance, method_name)
#             try:
#                 result = method(user_id)
#                 if isinstance(result, str):
#                     return json.loads(result)
#                 if isinstance(result, dict):
#                     return result
#             except Exception:
#                 continue
                
#     return {}


# def persist_safe_historical_context(store_instance, user_id: str, payload: dict) -> bool:
#     """
#     Dynamically maps execution payload write states into memory stores 
#     regardless of variations in contract execution naming signatures.
#     """
#     for method_name in ["save", "store", "persist", "set_context", "update_context"]:
#         if hasattr(store_instance, method_name):
#             method = getattr(store_instance, method_name)
#             try:
#                 # Try saving as raw dict object first
#                 method(user_id, payload)
#                 return True
#             except Exception:
#                 try:
#                     # Fallback to JSON serialized string if the table schemas require text primitives
#                     method(user_id, json.dumps(payload))
#                     return True
#                 except Exception:
#                     continue
#     return False


# def is_followup_query(prompt_lower: str) -> bool:
#     """Isolates context checks exclusively to explicit historical continuations."""
#     return any(re.search(term, prompt_lower) for term in FOLLOWUP_TERMS)


# def synthesize_ranking_weights(prompt_lower: str) -> tuple:
#     """
#     Replaced misleading confidence indicator values with accurate 
#     match_quality analytics flags based on target keyword precision.
#     """
#     base_weights = {
#         "price": 0.0, "area": 0.0, "amenities": 0.0,
#         "location": 0.0, "connectivity": 0.0, "distance": 0.0
#     }
    
#     intent_quality_logs = {}
    
#     for intent_name, keywords in RANKING_WORD_LISTS.items():
#         for keyword in keywords:
#             pattern = r"\b" + re.escape(keyword) + r"\b"
#             match = re.search(pattern, prompt_lower)
            
#             if match:
#                 start_idx = match.start()
#                 preceding_chunk = prompt_lower[max(0, start_idx - 30):start_idx].strip()
                
#                 if any(re.search(neg, preceding_chunk) for neg in NEGATIONS):
#                     continue 
                
#                 strength_score = 1.0
#                 for modifier, multiplier in INTENSITY_MODIFIERS.items():
#                     occurrences = len(re.findall(r"\b" + re.escape(modifier) + r"\b", preceding_chunk))
#                     if occurrences > 0:
#                         strength_score += (multiplier - 1.0) * occurrences
                
#                 quality_metric = 0.95 if keyword in ["low budget", "luxury", "spacious", "metro"] else 0.85
                
#                 intent_quality_logs[intent_name] = {
#                     "intent": intent_name,
#                     "strength": round(strength_score, 2),
#                     "match_quality": quality_metric
#                 }
#                 break 

#     for intent_name, metrics in intent_quality_logs.items():
#         target_map = RANKING_TARGET_MAPS.get(intent_name, {})
#         for feature, feature_weight in target_map.items():
#             if feature in base_weights:
#                 base_weights[feature] += (metrics["strength"] * feature_weight)

#     return base_weights, intent_quality_logs


# def parse_intent_and_execute(user_prompt: str, session_state_tray: list, current_ui_sliders: dict = None, user_changed_sliders: bool = False) -> dict:
#     """
#     Main entry point executing structural search filters alongside ranking preferences.
#     """
#     prompt_lower = user_prompt.lower().strip()

#     # -----------------------------------------------------------------
#     # STEP 1: RESOLVE HISTORICAL AMNESTY BOUNDARIES (FIXED RETRIEVAL)
#     # -----------------------------------------------------------------
#     historical_context = fetch_safe_historical_context(memory_store, USER_ID)
    
#     if is_followup_query(prompt_lower):
#         historical_filters = historical_context.get("filters", {})
#         historical_weights = historical_context.get("weights", {})
#     else:
#         historical_filters = {}
#         historical_weights = {}

#     # -----------------------------------------------------------------
#     # STEP 2: ROUTE AGENT METRIC ACTIONS
#     # -----------------------------------------------------------------
#     if any(k in prompt_lower for k in ["compare", "ranking", "rank"]):
#         if len(session_state_tray) < 2:
#             return {"type": "text", "content": "⚠️ I need at least 2 properties in your tray to run an investment comparison."}
#         print("✅  mcp compare get trigger")
#         return {"type": "comparison", "content": tools.compare_properties(session_state_tray)}

#     if any(k in prompt_lower for k in ["rent", "rental", "tenant", "lease", "yield", "rental yield", "monthly rent"]):
#         if len(session_state_tray) < 1:
#             return {"type": "text", "content": "⚠️ Please add at least one target property to your evaluation tray first."}
#         return {"type": "rental", "content": tools.get_rental_analysis(session_state_tray)}

#     if any(k in prompt_lower for k in ["predict", "prediction", "predicted price", "estimated price"]):
#         if len(session_state_tray) < 1:
#             return {"type": "text", "content": "⚠️ Your evaluation tray is empty. Stage properties to run predictions."}
#         return {"type": "prediction", "content": tools.get_price_prediction(session_state_tray)}

#     # -----------------------------------------------------------------
#     # STEP 3: SEPARATED STRUCTURAL FILTER EXTRACTION
#     # -----------------------------------------------------------------
#     extracted_filters = {"bhk": None, "amenities": None, "location": None}

#     bhk_match = re.search(FILTER_INTENTS["bhk_pattern"], prompt_lower)
#     if bhk_match:
#         extracted_filters["bhk"] = f"{bhk_match.group(1)}bhk"
#     else:
#         extracted_filters["bhk"] = historical_filters.get("bhk")

#     matched_locations = []
#     known_locations = CACHED_SEARCH_METADATA.get("location", [])
#     for loc in known_locations:
#         if re.search(r"\b" + re.escape(str(loc).lower()) + r"\b", prompt_lower):
#             matched_locations.append(loc)

#     if matched_locations:
#         extracted_filters["location"] = " ".join(matched_locations)
#     else:
#         extracted_filters["location"] = historical_filters.get("location")

#     matched_amenities = []
#     known_amenities = CACHED_SEARCH_METADATA.get("amenities_mcp", [])
#     for amenity in known_amenities:
#         if re.search(r"\b" + re.escape(str(amenity).lower()) + r"\b", prompt_lower):
#             matched_amenities.append(amenity)

#     if matched_amenities:
#         extracted_filters["amenities"] = " ".join(matched_amenities)
#     else:
#         extracted_filters["amenities"] = historical_filters.get("amenities")

#     # -----------------------------------------------------------------
#     # STEP 4: SEPARATED PREFERENCE EXTRACTION & BLENDED EVALUATION
#     # -----------------------------------------------------------------
#     if extracted_filters["location"] or extracted_filters["amenities"] or extracted_filters["bhk"]:
        
#         raw_results = tools.search_properties(
#             bhk=extracted_filters["bhk"],
#             amenities=extracted_filters["amenities"],
#             location=extracted_filters["location"],
#             limit=30
#         )
        
#         if raw_results:
#             results_df = pd.DataFrame(raw_results)
            
#             matched_full_df = GLOBAL_MASTER_DF[GLOBAL_MASTER_DF["id"].isin(results_df["id"])].copy()
#             matched_full_df = matched_full_df.merge(results_df[["id", "search_score"]], on="id", how="left")
#             matched_full_df = matched_full_df.rename(columns={"search_score": "cosine_similarity"})

#             # Generate weights vectors alongside tracking metadata
#             synthesized_chat_weights, quality_metadata = synthesize_ranking_weights(prompt_lower)
            
            
#             if sum(synthesized_chat_weights.values()) == 0 and historical_weights:
#                 synthesized_chat_weights = historical_weights

#             # =====================================================================
#             # PRODUCTION RUNTIME DEBUG TELEMETRY 
#             # =====================================================================
#             print("\n" + "="*50)
#             print("🔍 RUNTIME RANKING TELEMETRY (BEFORE HYBRID RANKING)")
#             print(f"INTENT WEIGHTS TYPE : {type(synthesized_chat_weights)}")
#             print(f"INTENT WEIGHTS RAW  : {synthesized_chat_weights}")
#             print(f"SLIDER WEIGHTS TYPE : {type(current_ui_sliders)}")
#             print(f"SLIDER WEIGHTS RAW  : {current_ui_sliders}")
#             print("="*50 + "\n")

#             # Execute unified ranker using true state tracking ratios
#             ranked_df = apply_hybrid_ranking(
#                 similar_df=matched_full_df, 
#                 intent_weights=synthesized_chat_weights, 
#                 slider_weights=current_ui_sliders, 
#                 alpha=0.65,
#                 user_changed_sliders=user_changed_sliders
#             )

#             # Record operational tracking history safely using multi-signature fallback interface
#             persist_safe_historical_context(
#                 store_instance=memory_store,
#                 user_id=USER_ID,
#                 payload={
#                     "query": user_prompt,
#                     "weights": synthesized_chat_weights,
#                     "filters": extracted_filters,
#                     "quality_logs": quality_metadata
#                 }
#             )

#             ranked_df = ranked_df.rename(columns={"hybrid_score": "search_score"})
#             ranked_df["amenities_mcp"] = ranked_df.get("amenities_mcp", "")
            
#             final_cols = ["id", "price", "bhk_type", "location", "amenities_mcp", "search_score", "why_recommended"]
#             display_cols = [c for c in final_cols if c in ranked_df.columns]
            
#             final_records = ranked_df[display_cols].head(5).to_dict(orient="records")

#             return {
#                 "type": "search_results",
#                 "content": final_records,
#                 "current_query_state": {
#                     "active_filters": extracted_filters,
#                     "chat_preference_weights": synthesized_chat_weights,
#                     "quality_metadata": quality_metadata
#                 }
#             }
#         else:
#             return {"type": "text", "content": f"❌ Zero properties matched infrastructure specifications: `{extracted_filters}`."}

#     # -----------------------------------------------------------------
#     # STEP 5: DEEPSEEK GENERIC CHAT FALLBACK
#     # -----------------------------------------------------------------
#     staged_context = GLOBAL_MASTER_DF[GLOBAL_MASTER_DF["id"].isin(session_state_tray)].head(3).to_string(index=False) if session_state_tray else "No active properties staged."
#     chat_prompt = f"""You are an expert real estate consultant. Answer the inquiry directly.

#     ACTIVE CONTEXT ROWS IN USER MEMORY TRAY:
#     {staged_context}

#     USER REQUEST INPUTS: {user_prompt}
#     Provide structured clear insights utilizing Indian Rupee (₹) denominations.
#     """
#     return {"type": "text", "content": ask_deepseek(chat_prompt)}


#==================================================================================================================================================================================================
#==================================================================================================================================================================================================
#==================================================================================================================================================================================================
#==================================================================================================================================================================================================
#==================================================================================================================================================================================================
#==================================================================================================================================================================================================
#==================================================================================================================================================================================================
#==================================================================================================================================================================================================
#==================================================================================================================================================================================================
#==================================================================================================================================================================================================

# # =====================================================================
# # chat_service.py (PRODUCTION ARCHITECTURE - UNIFIED SEMANTIC ENGINE)
# # =====================================================================

# import re
# import json
# import pandas as pd
# import streamlit as st

# import src.mcp.tools.property_tools as tools
# from src.core.search_registry import GLOBAL_MASTER_DF, CACHED_SEARCH_METADATA
# from src.llm.memory_store import SQLiteMemoryStore
# from src.llm.deepseek_client import ask_deepseek
# from src.recommender.hybrid_recommender import apply_hybrid_ranking

# memory_store = SQLiteMemoryStore()
# USER_ID = "default_user"

# # =====================================================================
# # CONFIGURATION MATRICES
# # =====================================================================

# # Structural hard constraints (used by Fallback Layer 2)
# FILTER_INTENTS = {
#     "bhk_pattern":      r"(\d+)\s*bhk",
#     "known_locations":  "location",
#     "known_amenities":  "amenities_mcp"
# }

# # Cleaned naming mapping dictionary to uppercase standard (Fallback Layer 2)
# RANKING_TARGET_MAPS = {
#     "low budget":      {"price": 1.0},
#     "luxury":          {"amenities": 1.0, "area": 0.8, "location": 0.4},
#     "spacious":        {"area": 1.0},
#     "good amenities":  {"amenities": 1.0},
#     "connectivity":    {"connectivity": 1.0, "distance": 0.6},
#     "location":        {"location": 1.0},
#     "investment":      {"price": 1.0, "location": 0.5},
#     "family":          {"location": 1.0, "area": 0.5}
# }

# RANKING_WORD_LISTS = {
#     "low budget":      ["low budget", "cheap", "affordable", "under budget", "value for money", "pocket friendly", "lowest price", "pocket-friendly", "economical"],
#     "luxury":          ["luxury", "premium", "posh", "high end", "elite", "luxurious", "expensive", "high-end", "ultra-premium"],
#     "spacious":        ["spacious", "big size", "large area", "huge", "roomy", "bigger rooms", "carpet area", "massive square feet"],
#     "good amenities":  ["good amenities", "premium community features", "high-end facilities"],
#     "connectivity":    ["near station", "metro", "railway", "connectivity", "public transport", "highway", "link road", "walkable", "easy commuting", "commuting is easier", "access to office", "less travel time", "save travel time"],
#     "location":        ["great location", "prime location", "well located", "center of city", "heart of mumbai"],
#     "investment":      ["investment", "good yield", "resale value", "future growth", "high return", "roi", "appreciation"],
#     "family":          ["family oriented", "safe neighborhood", "school proximity", "gated community"]
# }

# # Advanced Modifier Token Layers (Fallback Layer 2)
# NEGATIONS = [r"not\b", r"don't\b", r"do\s+not\b", r"without\b", r"avoid\b", r"no\b", r"never\b"]

# INTENSITY_MODIFIERS = {
#     "extremely": 2.0, "super": 1.5, "very": 1.4, "highly": 1.4,
#     "good": 1.1, "great": 1.2, "prime": 1.2, "ultra": 2.0
# }

# # Followup Continuity Tracking Layers
# FOLLOWUP_TERMS = [

#     r"similar\b",
#     r"show\s+more\b",

#     r"another\b",
#     r"different\b",
#     r"alternative\b",
#     r"next\b",

#     r"cheaper\b",
#     r"better\b",
#     r"closer\b",
#     r"nearer\b",

#     r"other\s+options\b",
#     r"more\s+options\b",

#     r"also\b",
#     r"above\b",

# ]

# # =====================================================================
# # SEMANTIC IMPORTANCE CONFIGURATION LAYER
# # =====================================================================

# # Semantic importance value mapping to standard weight floats
# IMPORTANCE_TO_WEIGHT = {
#     "very_high": 1.5,
#     "high":      1.0,
#     "medium":    0.5,
#     "low":       0.2,
#     "none":      0.0
# }

# # Master fallback default matrix for unmapped or empty inferences
# DEFAULT_BLANK_WEIGHTS = {
#     "price": 0.0, "area": 0.0, "amenities": 0.0,
#     "location": 0.0, "connectivity": 0.0, "distance": 0.0
# }

# # =====================================================================
# # SAFE MEMORY INTERACTION LAYER
# # =====================================================================

# def fetch_safe_historical_context(store_instance, user_id: str) -> dict:

#     """Fetches the user's previous filters, preferences, and search history from memory."""

#     for method_name in ["get_latest_context", "load", "get_context", "get_history"]:
#         if hasattr(store_instance, method_name):
#             method = getattr(store_instance, method_name)
#             try:
#                 result = method(user_id)
#                 if isinstance(result, str):
#                     return json.loads(result)
#                 if isinstance(result, dict):
#                     return result
#             except Exception:
#                 continue

#     return {}


# def persist_safe_historical_context(store_instance, user_id: str, payload: dict) -> bool:
#     """
#     Saves the user's search filters and preferences so they can
#     be reused in follow-up conversations.

#     Returns:
#         bool: True if saved successfully, otherwise False.
#     """
#     for method_name in ["save", "store", "persist", "set_context", "update_context"]:
#         if hasattr(store_instance, method_name):
#             method = getattr(store_instance, method_name)
#             try:
#                 # Try saving as raw dict object first
#                 method(user_id, payload)
#                 return True
#             except Exception:
#                 try:
#                     # Fallback to JSON serialized string if the table schemas require text primitives
#                     method(user_id, json.dumps(payload))
#                     return True
#                 except Exception:
#                     continue
#     return False


# def is_followup_query(prompt_lower: str) -> bool:
#     """
#     Checks if the user's query contains follow-up keywords and returns True or False.

#     example:
#     User: "show me more options"
#     → True

#     User: "what about in Thane?"
#     → True

#     User: "find 2 BHK in Mumbai"
#     → False
#     """
#     return any(re.search(term, prompt_lower) for term in FOLLOWUP_TERMS)


# # =====================================================================
# # HIGH-FIDELITY RECOMMENDATION GENERATOR
# # =====================================================================

# def generate_custom_recommendation_reason(row: pd.Series, preferences: dict) -> str:
#     """
#     Creates a personalized explanation describing why a property
#     was recommended based on the user's preferences.
#     """
#     reasons = []

#     price_pref = preferences.get("price_importance", "none")
#     amenities_pref = preferences.get("amenities_importance", "none")
#     connectivity_pref = preferences.get("connectivity_importance", "none")
#     area_pref = preferences.get("area_importance", "none")
#     location_pref = preferences.get("location_importance", "none")

#     # 1. Budget Preference Evaluation
#     if price_pref in ["high", "very_high"]:
#         reasons.append("aligns closely with your goal of finding budget-friendly, affordable housing")

#     # 2. Amenities Preference Evaluation
#     if amenities_pref in ["high", "very_high"] and hasattr(row, "amenities_mcp") and str(row.amenities_mcp).strip():
#         reasons.append("provides high-quality localized community lifestyle facilities and modern amenities")

#     # 3. Connectivity Preference Evaluation
#     if connectivity_pref in ["high", "very_high"]:
#         reasons.append("offers strategic proximity to transit networks, simplifying daily commutes")

#     # 4. Spatial Preference Evaluation
#     if area_pref in ["high", "very_high"]:
#         reasons.append("features spacious layout plans with larger, more generous carpet areas")

#     # 5. Elite/Prime Location Preference Evaluation
#     if location_pref in ["high", "very_high"]:
#         reasons.append("positions you in a premium, highly coveted, and well-located neighborhood")

#     # Synthesize tailored descriptions securely
#     if reasons:
#         cleaned_reasons = []
#         for r in reasons:
#             r_clean = r.strip()
#             # Format and adjust uppercase letters if we have chained reasoning phrases
#             if cleaned_reasons and r_clean and r_clean[0].isupper():
#                 r_clean = r_clean[0].lower() + r_clean[1:]
#             cleaned_reasons.append(r_clean)

#         explanation = "This property is highly recommended because it " + ", and it ".join(cleaned_reasons) + f" located in {row.get('location', 'Mumbai')}."
#     else:
#         explanation = f"Matches your parameters in {row.get('location', 'Mumbai')} with a competitive price and standard features."

#     return explanation


# # =====================================================================
# # DUAL-LAYER UNIFIED INTENT EXTRACTOR (LLM + REGEX FALLBACK)
# # =====================================================================

# def extract_intent_and_preferences(user_prompt: str, historical_filters: dict = None, historical_weights: dict = None) -> dict:
#     """
#     Extracts property search filters and preferences from a user query.

#     The function first uses an LLM to identify:
#     - Filters: BHK, location, amenities
#     - Preferences: price, area, connectivity, location, amenities

#     If the LLM fails, it falls back to regex-based extraction.

#     Preferences are converted into numerical weights that can be used
#     by the property recommendation/ranking engine.

#     Follow-up queries can reuse previously detected filters and weights.

#     Args:
#         user_prompt (str): User's property search query.
#         historical_filters (dict, optional): Filters from previous queries.
#         historical_weights (dict, optional): Weights from previous queries.

#     Returns:
#         dict: Extracted filters, preferences, ranking weights,
#         and parsing source information.
#     """
#     prompt_lower = user_prompt.lower().strip()
#     historical_filters = historical_filters or {}
#     historical_weights = historical_weights or {}

#     # -----------------------------------------------------------------
#     # LAYER 1: LLM SEMANTIC INTENT EXTRACTION
#     # -----------------------------------------------------------------
#     system_parsing_instruction = """You are an advanced real estate semantic interpretation engine.
# Your task is to parse user queries into HARD CONSTRAINTS (strict filters) and SOFT PREFERENCES (ranking weights).

# Return EXACTLY a single JSON object matching this schema, without any conversational preamble or Markdown wraps.

# {
#   "filters": {
#     "bhk": "Ex: '2bhk', '3bhk' or null if not explicitly mentioned",
#     "location": "Ex: 'Thane', 'Andheri' or null if no location specified",
#     "amenities": "Ex: 'gym, pool', 'clubhouse' or null if no specific facilities are requested"
#   },
#   "preferences": {
#     "price_importance": "Set to 'high'/'very_high' if user wants cheap, affordable, low cost, or pocket-friendly. Otherwise 'none'/'low'/'medium'",
#     "amenities_importance": "Set to 'high'/'very_high' if they prioritize facilities like gym, pool, security, clubhouse. Otherwise 'none'/'low'/'medium'",
#     "location_importance": "Allowed values: ['very_high', 'high', 'medium', 'low', 'none']. Use: 'none' for a normal location mention, 'medium' if a good location is preferred, 'high' for a prime/premium location, and 'very_high' for an elite/ultra-premium location",
#     "connectivity_importance": "Set to 'high'/'very_high' if commuting near a metro, station, highway or link road is prioritized. Otherwise 'none'/'low'/'medium'",
#     "area_importance": "Set to 'high'/'very_high' if they seek big rooms, spacious size, or huge carpet areas. Otherwise 'none'/'low'/'medium'"
#   }
# }

# IMPORTANT LOCATION RULE (PREVENT DOUBLE-COUNTING):
# If a user merely specifies a geographic location (e.g. 'Thane', 'Andheri', 'Powai', 'Navi Mumbai', 'Kandivali') without describing it with terms like 'prime', 'heart of city', 'posh area', or 'great central spot':
# 1. Map that location value strictly into filters.location
# 2. Set preferences.location_importance to 'none'
# Set preferences.location_importance to 'high' or 'very_high' ONLY if they are explicitly demanding high-prestige, premium, elite, or ultra-central geographic placements.
# """

#     llm_payload_prompt = f"{system_parsing_instruction}\n\nUSER REQUEST: {user_prompt}\nJSON OUTPUT:"

#     try:
#         llm_raw_response = ask_deepseek(
#             llm_payload_prompt
#         ).strip()

#         # Clean potential markdown formatting
#         if llm_raw_response.startswith("```"):

#             llm_raw_response = re.sub(
#                 r"^```(?:json)?\s*",
#                 "",
#                 llm_raw_response
#             )

#             llm_raw_response = re.sub(
#                 r"\s*```$",
#                 "",
#                 llm_raw_response
#             )

#         json_match = re.search(r"\{.*\}", llm_raw_response, re.DOTALL)
#         if json_match:
#             llm_raw_response = json_match.group(0)
            
#         parsed_data = json.loads(llm_raw_response)

#         print("\n===== LLM RAW RESPONSE =====")
#         print(llm_raw_response)
#         print("============================\n")

#         filters = parsed_data.get(
#             "filters",
#             {"bhk": None, "location": None, "amenities": None}
#         )

#         bhk_match = re.search(
#             r"(\d+)\s*bhk",
#             prompt_lower
#         )

#         if bhk_match:
#             filters["bhk"] = f"{bhk_match.group(1)}bhk"

#         # ==================================================
#         # LOCATION FALLBACK USING METADATA
#         # ==================================================
#         if not filters.get("location"):

#             known_locations = CACHED_SEARCH_METADATA.get("location", [])

#             for loc in known_locations:

#                 if pd.isna(loc):
#                     continue

#                 loc_lower = str(loc).lower().strip()

#                 if loc_lower and re.search(
#                     r"\b" + re.escape(loc_lower) + r"\b",
#                     prompt_lower
#                 ):
#                     filters["location"] = loc
#                     break

#         # ==================================================
#         # CITY FALLBACK
#         # ==================================================
#         if not filters.get("location"):

#             known_cities = ["mumbai", "thane", "navi mumbai", "palghar"]

#             for city in known_cities:

#                 if re.search(
#                     r"\b" + re.escape(city) + r"\b",
#                     prompt_lower
#                 ):
#                     filters["location"] = city.title()
#                     break

#         preferences = parsed_data.get("preferences", {})

#         print("\n===== LOCATION FALLBACK =====")
#         print("Detected Location:", filters.get("location"))
#         print("=============================\n")

#         # ==================================================
#         # AMENITIES PREFERENCE FALLBACK
#         # ==================================================

#         # Generic amenities intent
#         if any(
#             word in prompt_lower
#             for word in [
#                 "amenities",
#                 "facility",
#                 "facilities"
#             ]
#         ):
#             preferences["amenities_importance"] = "high"

#         # Specific amenity names from metadata
#         known_amenities = CACHED_SEARCH_METADATA.get(
#             "amenities_mcp",
#             []
#         )

#         for amenity in known_amenities:

#             if pd.isna(amenity):
#                 continue

#             amenity_text = str(amenity).lower().strip()

#             if amenity_text and amenity_text in prompt_lower:
#                 preferences["amenities_importance"] = "high"
#                 break

#         # Map preference importances directly into numerical weights
#         synthesized_weights = {
#             "price":        IMPORTANCE_TO_WEIGHT.get(preferences.get("price_importance"), 0.0),
#             "amenities":    IMPORTANCE_TO_WEIGHT.get(preferences.get("amenities_importance"), 0.0),
#             "location":     IMPORTANCE_TO_WEIGHT.get(preferences.get("location_importance"), 0.0),
#             "connectivity": IMPORTANCE_TO_WEIGHT.get(preferences.get("connectivity_importance"), 0.0),
#             "area":         IMPORTANCE_TO_WEIGHT.get(preferences.get("area_importance"), 0.0),
#             "distance":     0.6 if preferences.get("connectivity_importance") in ["high", "very_high"] else 0.0
#         }

#         print("\n===== AMENITIES FALLBACK =====")
#         print("Amenities Importance:", preferences.get("amenities_importance"))
#         print("Amenities Weight:", synthesized_weights["amenities"])
#         print("==============================\n")
        
#         # Handle followup logic integration
#         if is_followup_query(prompt_lower):
#             for k, v in historical_filters.items():
#                 if not filters.get(k):
#                     filters[k] = v
#             if sum(synthesized_weights.values()) == 0:
#                 synthesized_weights = historical_weights
                
#         return {
#             "filters": filters,
#             "preferences": preferences,
#             "weights": synthesized_weights,
#             "source": "llm_unified_parser"
#         }
        
#     except Exception as e:
#         print(f"⚠️ Layer 1 LLM Unified Parsing Exception: {str(e)}. Defaulting to Layer 2 Regex Heuristics.")

#     # -----------------------------------------------------------------
#     # LAYER 2: DETERMINISTIC REGEX HEURISTIC FALLBACK
#     # -----------------------------------------------------------------
#     extracted_filters = {"bhk": None, "amenities": None, "location": None}

#     # Extract BHK constraints
#     bhk_match = re.search(FILTER_INTENTS["bhk_pattern"], prompt_lower)
#     if bhk_match:
#         extracted_filters["bhk"] = f"{bhk_match.group(1)}bhk"
#     else:
#         extracted_filters["bhk"] = historical_filters.get("bhk")

#     # Extract Location constraints using master taxonomy metadata
#     matched_locations = []
#     known_locations = CACHED_SEARCH_METADATA.get("location", [])
#     for loc in known_locations:
#         if re.search(r"\b" + re.escape(str(loc).lower()) + r"\b", prompt_lower):
#             matched_locations.append(loc)
#     if matched_locations:
#         extracted_filters["location"] = " ".join(matched_locations)
#     else:
#         extracted_filters["location"] = historical_filters.get("location")

#     # Extract Amenity constraints using master taxonomy metadata
#     matched_amenities = []
#     known_amenities = CACHED_SEARCH_METADATA.get("amenities_mcp", [])
#     for amenity in known_amenities:
#         if re.search(r"\b" + re.escape(str(amenity).lower()) + r"\b", prompt_lower):
#             matched_amenities.append(amenity)
#     if matched_amenities:
#         extracted_filters["amenities"] = " ".join(matched_amenities)
#     else:
#         extracted_filters["amenities"] = historical_filters.get("amenities")

#     # Synthesize preferences via Regex mappings
#     base_weights = DEFAULT_BLANK_WEIGHTS.copy()
#     intent_quality_logs = {}
    
#     for intent_name, keywords in RANKING_WORD_LISTS.items():
#         for keyword in keywords:
#             pattern = r"\b" + re.escape(keyword) + r"\b"
#             match = re.search(pattern, prompt_lower)
            
#             if match:
#                 start_idx = match.start()
#                 preceding_chunk = prompt_lower[max(0, start_idx - 30):start_idx].strip()
                
#                 if any(re.search(neg, preceding_chunk) for neg in NEGATIONS):
#                     continue 
                
#                 strength_score = 1.0
#                 for modifier, multiplier in INTENSITY_MODIFIERS.items():
#                     occurrences = len(re.findall(r"\b" + re.escape(modifier) + r"\b", preceding_chunk))
#                     if occurrences > 0:
#                         strength_score += (multiplier - 1.0) * occurrences
                
#                 quality_metric = 0.95 if keyword in ["low budget", "luxury", "spacious", "metro"] else 0.85
                
#                 intent_quality_logs[intent_name] = {
#                     "intent": intent_name,
#                     "strength": round(strength_score, 2),
#                     "match_quality": quality_metric,
#                     "source": "regex_keyword_fallback"
#                 }
#                 break 

#     for intent_name, metrics in intent_quality_logs.items():
#         target_map = RANKING_TARGET_MAPS.get(intent_name, {})
#         for feature, feature_weight in target_map.items():
#             if feature in base_weights:
#                 base_weights[feature] += (metrics["strength"] * feature_weight)

#     # Reconstruct preferences format to mimic LLM layer structure
#     synthesized_preferences = {
#         "price_importance":        "high" if base_weights["price"] >= 1.0 else "none",
#         "amenities_importance":    "high" if base_weights["amenities"] >= 1.0 else "none",
#         "location_importance":     "high" if base_weights["location"] >= 1.0 else "none",
#         "connectivity_importance": "high" if base_weights["connectivity"] >= 1.0 else "none",
#         "area_importance":         "high" if base_weights["area"] >= 1.0 else "none"
#     }

#     return {
#         "filters": extracted_filters,
#         "preferences": synthesized_preferences,
#         "weights": base_weights,
#         "source": "regex_fallback"
#     }


# # =====================================================================
# # MAIN PIPELINE ENTRY POINT
# # =====================================================================

# def parse_intent_and_execute(user_prompt: str, session_state_tray: list, current_ui_sliders: dict = None) -> dict:
#     """
#     Main entry point executing structured search filters alongside ranking preferences.
#     """
#     prompt_lower = user_prompt.lower().strip()

#     print("\n===== USER QUERY =====")
#     print(user_prompt)
#     print("======================\n")

#     # -----------------------------------------------------------------
#     # STEP 1: RESOLVE HISTORICAL AMNESTY BOUNDARIES (FIXED RETRIEVAL)
#     # -----------------------------------------------------------------
#     historical_context = fetch_safe_historical_context(
#         memory_store,
#         USER_ID
#     )

#     is_followup = is_followup_query(prompt_lower)

#     print("\n===== FOLLOWUP DEBUG =====")
#     print("User Query:", user_prompt)
#     print("Detected Followup:", is_followup)
#     print("Historical Context:", historical_context)
#     print("==========================\n")

#     if is_followup:
#         historical_filters = historical_context.get("filters", {})
#         historical_weights = historical_context.get("weights", {})
#     else:
#         historical_filters = {}
#         historical_weights = {}

#     print("=========================================")
#     print("historical_filters",historical_filters)
#     print("historical_weights",historical_weights)
#     print("=========================================")

#     # -----------------------------------------------------------------
#     # STEP 2: ROUTE AGENT METRIC ACTIONS
#     # -----------------------------------------------------------------
#     if any(k in prompt_lower for k in ["compare", "ranking", "rank"]):
#         if len(session_state_tray) < 2:
#             return {"type": "text", "content": "⚠️ I need at least 2 properties in your tray to run an investment comparison."}
#         return {"type": "comparison", "content": tools.compare_properties(session_state_tray)}

#     if any(k in prompt_lower for k in ["rent", "rental", "tenant", "lease", "yield", "rental yield", "monthly rent"]):
#         if len(session_state_tray) < 1:
#             return {"type": "text", "content": "⚠️ Please add at least one target property to your evaluation tray first."}
#         return {"type": "rental", "content": tools.get_rental_analysis(session_state_tray)}

#     if any(k in prompt_lower for k in ["predict", "prediction", "predicted price", "estimated price"]):
#         if len(session_state_tray) < 1:
#             return {"type": "text", "content": "⚠️ Your evaluation tray is empty. Stage properties to run predictions."}
#         return {"type": "prediction", "content": tools.get_price_prediction(session_state_tray)}

#     # -----------------------------------------------------------------
#     # STEP 3: EXECUTE UNIFIED EXTRACTION (FILTERS + PREFERENCES)
#     # -----------------------------------------------------------------
#     extracted_intent = extract_intent_and_preferences(user_prompt, historical_filters, historical_weights)
#     extracted_filters = extracted_intent["filters"]
#     print("\n===== EXTRACTED INTENT =====")
#     print(extracted_intent)
#     print("============================\n")
#     synthesized_chat_weights = extracted_intent["weights"]
#     preferences = extracted_intent["preferences"]
#     extraction_source = extracted_intent["source"]

#     # -----------------------------------------------------------------
#     # STEP 4: BLENDED RETRIEVAL, RANKING, AND DYNAMIC EXPLANATION
#     # -----------------------------------------------------------------
#     if extracted_filters["location"] or extracted_filters["amenities"] or extracted_filters["bhk"]:

#         print("\n===== SEARCH PARAMETERS =====")
#         print("BHK      :", extracted_filters["bhk"])
#         print("Location :", extracted_filters["location"])
#         print("Amenities:", extracted_filters["amenities"])
#         print("=============================\n")
        
#         raw_results = tools.search_properties(
#             bhk=extracted_filters["bhk"],
#             amenities=extracted_filters["amenities"],
#             location=extracted_filters["location"],
#             limit=30
#         )
        
#         if raw_results:

#             print("\n===== RAW SEARCH RESULTS =====")
#             print("Count:", len(raw_results))

#             for r in raw_results[:5]:
#                 print(
#                     r.get("id"),
#                     r.get("location"),
#                     r.get("bhk_type")
#                 )

#             print("==============================\n")

#             results_df = pd.DataFrame(raw_results)
            
#             matched_full_df = GLOBAL_MASTER_DF[GLOBAL_MASTER_DF["id"].isin(results_df["id"])].copy()
#             matched_full_df = matched_full_df.merge(results_df[["id", "search_score"]], on="id", how="left")
#             matched_full_df = matched_full_df.rename(columns={"search_score": "cosine_similarity"})

#             # Ensure weights logic handles empty state fallbacks
#             if sum(synthesized_chat_weights.values()) == 0 and historical_weights:
#                 synthesized_chat_weights = historical_weights

#             # =====================================================================
#             # PRODUCTION RUNTIME DEBUG TELEMETRY 
#             # =====================================================================
#             print("\n" + "="*50)
#             print(f"🔍 RUNTIME RANKING TELEMETRY (SOURCE: {extraction_source.upper()})")
#             print(f"EXTRACTED FILTERS   : {extracted_filters}")
#             print(f"INTENT WEIGHTS RAW  : {synthesized_chat_weights}")
#             print(f"SLIDER WEIGHTS RAW  : {current_ui_sliders}")
#             print("="*50 + "\n")

#             # Execute unified ranker using calculated weights
#             ranked_df = apply_hybrid_ranking(
#                 similar_df=matched_full_df, 
#                 intent_weights=synthesized_chat_weights, 
#                 slider_weights=current_ui_sliders, 
#                 alpha=0.65,
#             )

#             # Programmatically inject naturalized recommendation reasons
#             ranked_df["why_recommended"] = ranked_df.apply(
#                 lambda row: generate_custom_recommendation_reason(row, preferences), axis=1
#             )

#             # Record operational tracking history safely inside SQLite store
#             persist_safe_historical_context(
#                 store_instance=memory_store,
#                 user_id=USER_ID,
#                 payload={
#                     "query": user_prompt,
#                     "weights": synthesized_chat_weights,
#                     "filters": extracted_filters,
#                     "preferences": preferences,
#                     "extraction_source": extraction_source
#                 }
#             )

#             ranked_df = ranked_df.rename(columns={"hybrid_score": "search_score"})
#             ranked_df["amenities_mcp"] = ranked_df.get("amenities_mcp", "")
            
#             final_cols = ["id", "price", "bhk_type", "location", "amenities_mcp", "search_score", "why_recommended"]
#             display_cols = [c for c in final_cols if c in ranked_df.columns]
            
#             final_records = ranked_df[display_cols].head(5).to_dict(orient="records")

#             return {
#                 "type": "search_results",
#                 "content": final_records,
#                 "current_query_state": {
#                     "active_filters": extracted_filters,
#                     "chat_preference_weights": synthesized_chat_weights,
#                     "preferences_telemetry": preferences,
#                     "extraction_source": extraction_source
#                 }
#             }
#         else:
#             return {"type": "text", "content": f"❌ Zero properties matched infrastructure specifications: `{extracted_filters}`."}

#     # -----------------------------------------------------------------
#     # STEP 5: DEEPSEEK GENERIC CHAT FALLBACK
#     # -----------------------------------------------------------------
#     staged_context = GLOBAL_MASTER_DF[GLOBAL_MASTER_DF["id"].isin(session_state_tray)].head(3).to_string(index=False) if session_state_tray else "No active properties staged."
#     chat_prompt = f"""You are an expert real estate consultant. Answer the inquiry directly.

#     ACTIVE CONTEXT ROWS IN USER MEMORY TRAY:
#     {staged_context}

#     USER REQUEST INPUTS: {user_prompt}
#     Provide structured clear insights utilizing Indian Rupee (₹) denominations.
#     """
#     return {"type": "text", "content": ask_deepseek(chat_prompt)}


#===============================================================================================================================================================================

# =====================================================================
# chat_service.py (PRODUCTION ARCHITECTURE - UNIFIED SEMANTIC ENGINE)
# =====================================================================

import re
import json
import pandas as pd
import streamlit as st

import src.mcp.tools.property_tools as tools
from src.core.search_registry import GLOBAL_MASTER_DF, CACHED_SEARCH_METADATA
from src.llm.deepseek_client import ask_deepseek
from src.recommender.hybrid_recommender import apply_hybrid_ranking


#what this functions are return than mention at the end 

# =====================================================================
# CONFIGURATION MATRICES
# =====================================================================

# Structural hard constraints (used by Fallback Layer 2)
FILTER_INTENTS = {
    "bhk_pattern":      r"(\d+)\s*bhk",
    "known_locations":  "location",
    "known_amenities":  "amenities_mcp"
}

# Cleaned naming mapping dictionary to uppercase standard (Fallback Layer 2)
RANKING_TARGET_MAPS = {
    "low budget":      {"price": 1.0},
    "luxury":          {"amenities": 1.0, "area": 0.8, "location": 0.4},
    "spacious":        {"area": 1.0},
    "good amenities":  {"amenities": 1.0},
    "connectivity":    {"connectivity": 1.0, "distance": 0.6},
    "location":        {"location": 1.0},
    "investment":      {"price": 1.0, "location": 0.5},
    "family":          {"location": 1.0, "area": 0.5}
}

RANKING_WORD_LISTS = {
    "low budget":      ["low budget", "cheap", "affordable", "under budget", "value for money", "pocket friendly", "lowest price", "pocket-friendly", "economical"],
    "luxury":          ["luxury", "premium", "posh", "high end", "elite", "luxurious", "expensive", "high-end", "ultra-premium"],
    "spacious":        ["spacious", "big size", "large area", "huge", "roomy", "bigger rooms", "carpet area", "massive square feet"],
    "good amenities":  ["good amenities", "premium community features", "high-end facilities"],
    "connectivity":    ["near station", "metro", "railway", "connectivity", "public transport", "highway", "link road", "walkable", "easy commuting", "commuting is easier", "access to office", "less travel time", "save travel time"],
    "location":        ["great location", "prime location", "well located", "center of city", "heart of mumbai"],
    "investment":      ["investment", "good yield", "resale value", "future growth", "high return", "roi", "appreciation"],
    "family":          ["family oriented", "safe neighborhood", "school proximity", "gated community"]
}

# Advanced Modifier Token Layers (Fallback Layer 2)
NEGATIONS = [r"not\b", r"don't\b", r"do\s+not\b", r"without\b", r"avoid\b", r"no\b", r"never\b"]

INTENSITY_MODIFIERS = {
    "extremely": 2.0, "super": 1.5, "very": 1.4, "highly": 1.4,
    "good": 1.1, "great": 1.2, "prime": 1.2, "ultra": 2.0
}

# Followup Continuity Tracking Layers
FOLLOWUP_TERMS = [

    r"similar\b",
    r"show\s+more\b",

    r"\bmore\b",

    r"another\b",
    r"different\b",
    r"alternative\b",
    r"next\b",

    r"cheaper\b",
    r"better\b",
    r"closer\b",
    r"nearer\b",

    r"other\s+options\b",
    r"more\s+options\b",

    r"also\b",
    r"above\b",

]

# =====================================================================
# SEMANTIC IMPORTANCE CONFIGURATION LAYER
# =====================================================================

# Semantic importance value mapping to standard weight floats
IMPORTANCE_TO_WEIGHT = {
    "very_high": 1.5,
    "high":      1.0,
    "medium":    0.5,
    "low":       0.2,
    "none":      0.0
}

# Master fallback default matrix for unmapped or empty inferences
DEFAULT_BLANK_WEIGHTS = {
    "price": 0.0, "area": 0.0, "amenities": 0.0,
    "location": 0.0, "connectivity": 0.0, "distance": 0.0
}


def is_followup_query(prompt_lower: str) -> bool:
    """
    Detects whether the current query is a continuation
    of the previous property search.

    Examples:
        "show more"                  -> True
        "more options"               -> True
        "another property"           -> True
        "different properties"       -> True
        "cheaper ones"               -> True
        "find 2 bhk in thane"        -> False
        "show 3 bhk in navi mumbai"  -> False

    Returns:
        bool: True if the query should reuse the previous
        search filters and pagination state.
    """
    print("☑️ is_followup_query executed")
    matched = any(
        re.search(term, prompt_lower)
        for term in FOLLOWUP_TERMS
    )

    print(
        f"FOLLOWUP_CHECK={matched} "
        f"QUERY='{prompt_lower}'"
    )

    return matched


# =====================================================================
# HIGH-FIDELITY RECOMMENDATION GENERATOR
# =====================================================================

def generate_custom_recommendation_reason(row: pd.Series, preferences: dict) -> str:
    """
    Creates a personalized explanation describing why a property
    was recommended based on the user's preferences.
    """
    print("☑️ generate_custom_recommendation_reason executed")
    reasons = []

    price_pref = preferences.get("price_importance", "none")
    amenities_pref = preferences.get("amenities_importance", "none")
    connectivity_pref = preferences.get("connectivity_importance", "none")
    area_pref = preferences.get("area_importance", "none")
    location_pref = preferences.get("location_importance", "none")

    # 1. Budget Preference Evaluation
    if price_pref in ["high", "very_high"]:
        reasons.append("aligns closely with your goal of finding budget-friendly, affordable housing")

    # 2. Amenities Preference Evaluation
    if amenities_pref in ["high", "very_high"] and hasattr(row, "amenities_mcp") and str(row.amenities_mcp).strip():
        reasons.append("provides high-quality localized community lifestyle facilities and modern amenities")

    # 3. Connectivity Preference Evaluation
    if connectivity_pref in ["high", "very_high"]:
        reasons.append("offers strategic proximity to transit networks, simplifying daily commutes")

    # 4. Spatial Preference Evaluation
    if area_pref in ["high", "very_high"]:
        reasons.append("features spacious layout plans with larger, more generous carpet areas")

    # 5. Elite/Prime Location Preference Evaluation
    if location_pref in ["high", "very_high"]:
        reasons.append("positions you in a premium, highly coveted, and well-located neighborhood")

    # Synthesize tailored descriptions securely
    if reasons:
        cleaned_reasons = []
        for r in reasons:
            r_clean = r.strip()
            # Format and adjust uppercase letters if we have chained reasoning phrases
            if cleaned_reasons and r_clean and r_clean[0].isupper():
                r_clean = r_clean[0].lower() + r_clean[1:]
            cleaned_reasons.append(r_clean)

        explanation = "This property is highly recommended because it " + ", and it ".join(cleaned_reasons) + f" located in {row.get('location', 'Mumbai')}."
    else:
        explanation = f"Matches your parameters in {row.get('location', 'Mumbai')} with a competitive price and standard features."

    return explanation


# =====================================================================
# DUAL-LAYER UNIFIED INTENT EXTRACTOR (LLM + REGEX FALLBACK)
# =====================================================================

def extract_intent_and_preferences(user_prompt: str, historical_filters: dict = None, historical_weights: dict = None) -> dict:
    """
    Extracts property search filters and preferences from a user query.

    The function first uses an LLM to identify:
    - Filters: BHK, location, amenities
    - Preferences: price, area, connectivity, location, amenities

    If the LLM fails, it falls back to regex-based extraction.

    Preferences are converted into numerical weights that can be used
    by the property recommendation/ranking engine.

    Follow-up queries can reuse previously detected filters and weights.

    Args:
        user_prompt (str): User's property search query.
        historical_filters (dict, optional): Filters from previous queries.
        historical_weights (dict, optional): Weights from previous queries.

    Returns:
        dict: Extracted filters, preferences, ranking weights,
        and parsing source information.
    """
    print("☑️ extract_intent_and_preferences executed")
    prompt_lower = user_prompt.lower().strip()
    historical_filters = historical_filters or {}
    historical_weights = historical_weights or {}

    print("\n===== USER QUERY =====")
    print("extract_intent_and_preferences", prompt_lower)
    print("======================\n")

    # -----------------------------------------------------------------
    # LAYER 1: LLM SEMANTIC INTENT EXTRACTION
    # -----------------------------------------------------------------
    system_parsing_instruction = """You are an advanced real estate semantic interpretation engine.
Your task is to parse user queries into HARD CONSTRAINTS (strict filters) and SOFT PREFERENCES (ranking weights).

Return EXACTLY a single JSON object matching this schema, without any conversational preamble or Markdown wraps.

{
  "filters": {
    "bhk": "Ex: '2bhk', '3bhk' or null if not explicitly mentioned",
    "location": "Ex: 'Thane', 'Andheri' or null if no location specified",
    "amenities": "Ex: 'gym, pool', 'clubhouse' or null if no specific facilities are requested"
  },
  "preferences": {
    "price_importance": "Set to 'high'/'very_high' if user wants cheap, affordable, low cost, or pocket-friendly. Otherwise 'none'/'low'/'medium'",
    "amenities_importance": "Set to 'high'/'very_high' if they prioritize facilities like gym, pool, security, clubhouse. Otherwise 'none'/'low'/'medium'",
    "location_importance": "Allowed values: ['very_high', 'high', 'medium', 'low', 'none']. Use: 'none' for a normal location mention, 'medium' if a good location is preferred, 'high' for a prime/premium location, and 'very_high' for an elite/ultra-premium location",
    "connectivity_importance": "Set to 'high'/'very_high' if commuting near a metro, station, highway or link road is prioritized. Otherwise 'none'/'low'/'medium'",
    "area_importance": "Set to 'high'/'very_high' if they seek big rooms, spacious size, or huge carpet areas. Otherwise 'none'/'low'/'medium'"
  }
}

IMPORTANT LOCATION RULE (PREVENT DOUBLE-COUNTING):
If a user merely specifies a geographic location (e.g. 'Thane', 'Andheri', 'Powai', 'Navi Mumbai', 'Kandivali') without describing it with terms like 'prime', 'heart of city', 'posh area', or 'great central spot':
1. Map that location value strictly into filters.location
2. Set preferences.location_importance to 'none'
Set preferences.location_importance to 'high' or 'very_high' ONLY if they are explicitly demanding high-prestige, premium, elite, or ultra-central geographic placements.
"""

    llm_payload_prompt = f"{system_parsing_instruction}\n\nUSER REQUEST: {user_prompt}\nJSON OUTPUT:"
    
    #send the user query to the LLM, clean its response, convert it to a Python dictionary, and extract the filters.
    # Example final output:
    # {
    #     "filters": {
    #         "bhk": "3bhk",
    #         "location": "Thane",
    #         "amenities": "gym, swimming pool"
    #     },
    #     "preferences": {
    #         "price_importance": "high",
    #         "amenities_importance": "high",
    #         "location_importance": "none",
    #         "connectivity_importance": "none",
    #         "area_importance": "none"
    #     },
    #     "weights": {
    #         "price": 1.0,
    #         "amenities": 1.0,
    #         "location": 0.0,
    #         "connectivity": 0.0,
    #         "area": 0.0,
    #         "distance": 0.0
    #     },
    #     "source": "llm_unified_parser"
    # }
    try:
        llm_raw_response = ask_deepseek(
            llm_payload_prompt
        ).strip()

        # Clean potential markdown formatting
        if llm_raw_response.startswith("```"):

            llm_raw_response = re.sub(
                r"^```(?:json)?\s*",
                "",
                llm_raw_response
            )

            llm_raw_response = re.sub(
                r"\s*```$",
                "",
                llm_raw_response
            )

        json_match = re.search(r"\{.*\}", llm_raw_response, re.DOTALL)
        if json_match:
            llm_raw_response = json_match.group(0)
            
        parsed_data = json.loads(llm_raw_response)

        print("\n===== LLM RAW RESPONSE =====")
        print(llm_raw_response)
        print("============================\n")

        filters = parsed_data.get(
            "filters",
            {"bhk": None, "location": None, "amenities": None}
        )

        bhk_match = re.search(
            r"(\d+)\s*bhk",
            prompt_lower
        )

        if bhk_match:
            filters["bhk"] = f"{bhk_match.group(1)}bhk"

        # ==================================================
        # LOCATION FALLBACK USING METADATA
        # ==================================================
        # If the LLM misses the location, find it using known locations from search_metadata.json.
        
        # Note:
        # CACHED_SEARCH_METADATA is used only to recover the missing
        # location. The LLM has already extracted the other filters
        # and preferences successfully.
        if not filters.get("location"):

            known_locations = CACHED_SEARCH_METADATA.get("location", [])

            for loc in known_locations:

                if pd.isna(loc):
                    continue

                loc_lower = str(loc).lower().strip()

                if loc_lower and re.search(
                    r"\b" + re.escape(loc_lower) + r"\b",
                    prompt_lower
                ):
                    filters["location"] = loc
                    break

        # ==================================================
        # CITY FALLBACK
        # ==================================================
        #If the LLM couldn't find a location, this code checks whether the user mentioned one of the known cities (Mumbai, Thane, Navi Mumbai, Palghar) and uses it as the location.
        if not filters.get("location"):

            known_cities = ["mumbai", "thane", "navi mumbai", "palghar"]

            for city in known_cities:

                if re.search(
                    r"\b" + re.escape(city) + r"\b",
                    prompt_lower
                ):
                    filters["location"] = city.title()
                    break

        preferences = parsed_data.get("preferences", {})

        print("\n===== LOCATION FALLBACK =====")
        print("Detected Location:", filters.get("location"))
        print("=============================\n")

        # ==================================================
        # AMENITIES PREFERENCE FALLBACK
        # ==================================================
        # If the LLM misses the amenities preference, use
        # CACHED_SEARCH_METADATA to check whether the user
        # mentioned any known amenity in the query.

        # Generic amenities intent
        if any(word in prompt_lower for word in ["amenities", "facility", "facilities"]): # Backup: Detects that amenities are important if the user uses general words like 
                                                                                          # "amenities", "facility", or "facilities"  even if the LLM fails to identify this intent.
            preferences["amenities_importance"] = "high"

        #from the dictionary o/p get from the search_metadata.py from that fetch this amenities_mcp
        known_amenities = CACHED_SEARCH_METADATA.get("amenities_mcp",[])

        for amenity in known_amenities:
            if pd.isna(amenity): #Skip missing (NaN) values.
                continue

            amenity_text = str(amenity).lower().strip()

            if amenity_text and amenity_text in prompt_lower:
                preferences["amenities_importance"] = "high"  # If the user mentions a specific amenity, mark amenities as a high-priority preference.
                                                              # so here we get as example: - preferences = {"amenities_importance": "high"}
                break
        
        # ==================================================
        # PREFERENCE WEIGHT GENERATION
        # ==================================================
        # Map preference importances directly into numerical weights
        # get this numerical weights for that particular user query
        synthesized_weights = {
            "price":        IMPORTANCE_TO_WEIGHT.get(preferences.get("price_importance"), 0.0),
            "amenities":    IMPORTANCE_TO_WEIGHT.get(preferences.get("amenities_importance"), 0.0),
            "location":     IMPORTANCE_TO_WEIGHT.get(preferences.get("location_importance"), 0.0),
            "connectivity": IMPORTANCE_TO_WEIGHT.get(preferences.get("connectivity_importance"), 0.0),
            "area":         IMPORTANCE_TO_WEIGHT.get(preferences.get("area_importance"), 0.0),
            "distance":     0.6 if preferences.get("connectivity_importance") in ["high", "very_high"] else 0.0
        }

        print("\n===== AMENITIES FALLBACK =====")
        print("Amenities Importance:", preferences.get("amenities_importance"))
        print("Amenities Weight:", synthesized_weights["amenities"])
        print("==============================\n")
        
        # ==================================================
        # FOLLOW-UP QUERY HANDLING
        # ==================================================
        # If the current query is a follow-up, reuse the
        # previous filters and weights whenever needed.
        if is_followup_query(prompt_lower):
            for k, v in historical_filters.items():
                if not filters.get(k):
                    filters[k] = v
            if sum(synthesized_weights.values()) == 0:
                synthesized_weights = historical_weights

        # ==================================================
        # RETURN PARSED INTENT
        # ==================================================
        # Return the extracted filters, preferences,
        # numerical weights, and parsing source.                
        return {
            "filters": filters,
            "preferences": preferences,
            "weights": synthesized_weights,
            "source": "llm_unified_parser"
        }
    
    # ==================================================
    # LLM FAILURE → SWITCH TO REGEX FALLBACK
    # ==================================================
    # If the LLM returns invalid JSON or throws an error,
    # continue with the regex-based fallback parser.
    except Exception as e:
        print(f"⚠️ Layer 1 LLM Unified Parsing Exception: {str(e)}. Defaulting to Layer 2 Regex Heuristics.")


    # This entire block is Layer 2 (Fallback Parser). It only runs if the LLM fails (throws an exception or returns invalid JSON). 
    # Instead of using AI, it relies on regex patterns, keyword lists, and metadata to extract filters and estimate user preferences.

    # -----------------------------------------------------------------
    # LAYER 2: DETERMINISTIC REGEX HEURISTIC FALLBACK
    # -----------------------------------------------------------------
    extracted_filters = {"bhk": None, "amenities": None, "location": None} # Store extracted filters

    # -----------------------------
    # Extract BHK (e.g., 2BHK, 3BHK)
    # -----------------------------
    bhk_match = re.search(FILTER_INTENTS["bhk_pattern"], prompt_lower)
    if bhk_match:
        extracted_filters["bhk"] = f"{bhk_match.group(1)}bhk"
    else:
        extracted_filters["bhk"] = historical_filters.get("bhk") # Use previous BHK if none found

    # -----------------------------
    # Extract Location
    # -----------------------------
    # Note:
    # CACHED_SEARCH_METADATA is used again here, but this block runs
    # only when the LLM completely fails. In this case, the regex
    # fallback extracts all filters (BHK, location, amenities) from
    # the user query instead of relying on the LLM.
    matched_locations = []
    known_locations = CACHED_SEARCH_METADATA.get("location", [])

    # Check whether any known location appears in the user query
    for loc in known_locations:
        if re.search(r"\b" + re.escape(str(loc).lower()) + r"\b", prompt_lower):
            matched_locations.append(loc)
    if matched_locations:
        extracted_filters["location"] = " ".join(matched_locations)
    else:
        extracted_filters["location"] = historical_filters.get("location")

    # -----------------------------
    # Extract Amenities
    # -----------------------------
    matched_amenities = []
    known_amenities = CACHED_SEARCH_METADATA.get("amenities_mcp", [])

    # Check whether any known amenity appears in the user query
    for amenity in known_amenities:
        if re.search(r"\b" + re.escape(str(amenity).lower()) + r"\b", prompt_lower):
            matched_amenities.append(amenity)
    if matched_amenities:
        extracted_filters["amenities"] = " ".join(matched_amenities)
    else:
        extracted_filters["amenities"] = historical_filters.get("amenities") # Use previous amenities if none found

    # -----------------------------
    # Build preference weights
    # -----------------------------
    base_weights = DEFAULT_BLANK_WEIGHTS.copy()
    intent_quality_logs = {}

    # Look for ranking-related keywords
    for intent_name, keywords in RANKING_WORD_LISTS.items():
        for keyword in keywords:

            # Search keyword in the user query
            pattern = r"\b" + re.escape(keyword) + r"\b"
            match = re.search(pattern, prompt_lower)
            
            if match:
                # Get words before the matched keyword
                start_idx = match.start()
                preceding_chunk = prompt_lower[max(0, start_idx - 30):start_idx].strip()
                
                # Ignore if keyword is negated
                if any(re.search(neg, preceding_chunk) for neg in NEGATIONS):
                    continue 
                
                strength_score = 1.0 # Default keyword strength
                # Increase strength if modifiers like "very", "extremely" exist
                for modifier, multiplier in INTENSITY_MODIFIERS.items():
                    occurrences = len(re.findall(r"\b" + re.escape(modifier) + r"\b", preceding_chunk))
                    if occurrences > 0:
                        strength_score += (multiplier - 1.0) * occurrences
                
                # Assign confidence score
                quality_metric = 0.95 if keyword in ["low budget", "luxury", "spacious", "metro"] else 0.85
                
                # Save detected intent details
                intent_quality_logs[intent_name] = {
                    "intent": intent_name,
                    "strength": round(strength_score, 2),
                    "match_quality": quality_metric,
                    "source": "regex_keyword_fallback"
                }
                break 
    

    # Convert detected intents into feature weights
    for intent_name, metrics in intent_quality_logs.items():
        target_map = RANKING_TARGET_MAPS.get(intent_name, {})
        for feature, feature_weight in target_map.items():
            if feature in base_weights:
                base_weights[feature] += (metrics["strength"] * feature_weight)

    
    # -----------------------------
    # Convert numeric weights into
    # High / None preference labels
    # -----------------------------
    synthesized_preferences = {
        "price_importance":        "high" if base_weights["price"] >= 1.0 else "none",
        "amenities_importance":    "high" if base_weights["amenities"] >= 1.0 else "none",
        "location_importance":     "high" if base_weights["location"] >= 1.0 else "none",
        "connectivity_importance": "high" if base_weights["connectivity"] >= 1.0 else "none",
        "area_importance":         "high" if base_weights["area"] >= 1.0 else "none"
    }

    # Return extracted filters, preferences, and weights
    return {
        "filters": extracted_filters,
        "preferences": synthesized_preferences,
        "weights": base_weights,
        "source": "regex_fallback"
    }


# =====================================================================
# MAIN PIPELINE ENTRY POINT
# =====================================================================

def parse_intent_and_execute(user_prompt: str, session_state_tray: list, current_ui_sliders: dict = None) -> dict:
    """
    Main entry point executing structured search filters alongside ranking preferences.
    """
    print("☑️ parse_intent_and_execute executed")
    prompt_lower = user_prompt.lower().strip()

    print("\n===== USER QUERY =====")
    print(user_prompt)
    print("======================\n")

    # If the query is a follow-up (e.g. "show more", "another"), reuse previous search filters/weights
    # and move to the next page. Otherwise, start a fresh search from page 1 with empty history.
    is_followup = is_followup_query(prompt_lower)

    if is_followup:
        historical_filters = st.session_state.get(
            "last_search_filters",
            {}
        )

        historical_weights = st.session_state.get(
            "last_search_weights",
            {}
        )

        # Move to next page
        st.session_state["search_page"] = (
            st.session_state.get("search_page", 0) + 1
        )

    else:
        historical_filters = {}
        historical_weights = {}

        # New search starts from first page
        st.session_state["search_page"] = 0

    print("=========================================")
    print("historical_filters",historical_filters)
    print("historical_weights",historical_weights)
    print("=========================================")

    # -----------------------------------------------------------------
    # STEP 2: ROUTE AGENT METRIC ACTIONS
    # -----------------------------------------------------------------

    def require_tray(min_items=1):
        if len(session_state_tray) < min_items:
            return {
                "type": "text",
                "content": f"⚠️ Please add at least {min_items} property{'ies' if min_items > 1 else ''} to your evaluation tray first."
            }
        return None

    if any(k in prompt_lower for k in ["compare", "comparison", "ranking", "rank", "vs", "versus"]):
        error = require_tray(2)
        if error:
            return error
        return {"type": "comparison", "content": tools.compare_properties(session_state_tray)}

    if any(k in prompt_lower for k in ["rent", "rental", "tenant", "lease", "yield", "rental yield", "monthly rent"]):
        error = require_tray()
        if error:
            return error
        return {"type": "rental", "content": tools.get_rental_analysis(session_state_tray)}

    if any(k in prompt_lower for k in ["predict", "prediction", "predicted price", "estimated price", "price estimate", "property value"]):
        error = require_tray()
        if error:
            return error
        return {"type": "prediction", "content": tools.get_price_prediction(session_state_tray)}
    
    if any(k in prompt_lower for k in [ "negotiate", "negotiation", "discount", "deal", "target price", "bargain", "best price"]):
        error = require_tray()
        if error:
            return error
        return {"type": "negotiation", "content": tools.get_negotiation_strategy(session_state_tray)}

    if any(k in prompt_lower for k in ["valuation", "fair value", "fair price", "overpriced", "undervalued", "market value", "worth buying"]):
        error = require_tray()
        if error:
            return error
        return {"type": "valuation", "content": tools.get_valuation_analysis(session_state_tray)}

    if any(k in prompt_lower for k in ["should i buy", "buy decision", "investment advice", "best investment", "final advice"]):
        error = require_tray()
        if error:
            return error
        return {"type": "advisor", "content": tools.get_investment_advice(session_state_tray)}

    # -----------------------------------------------------------------
    # STEP 3: EXECUTE UNIFIED EXTRACTION (FILTERS + PREFERENCES)
    # -----------------------------------------------------------------
    extracted_intent = extract_intent_and_preferences(user_prompt, historical_filters, historical_weights)
    extracted_filters = extracted_intent["filters"]
    print("\n===== EXTRACTED INTENT =====")
    print(extracted_intent)
    print("============================\n")
    synthesized_chat_weights = extracted_intent["weights"]
    preferences = extracted_intent["preferences"]
    extraction_source = extracted_intent["source"]

    # -----------------------------------------------------------------
    # STEP 4: BLENDED RETRIEVAL, RANKING, AND DYNAMIC EXPLANATION
    # -----------------------------------------------------------------
    if extracted_filters["location"] or extracted_filters["amenities"] or extracted_filters["bhk"]:

        print("\n===== SEARCH PARAMETERS =====")
        print("BHK      :", extracted_filters["bhk"])
        print("Location :", extracted_filters["location"])
        print("Amenities:", extracted_filters["amenities"])
        print("=============================\n")
        
        raw_results = tools.search_properties(
            bhk=extracted_filters["bhk"],
            amenities=extracted_filters["amenities"],
            location=extracted_filters["location"],
            limit=30
        )
        
        if raw_results:

            print("\n===== RAW SEARCH RESULTS =====")
            print("Count:", len(raw_results))

            for r in raw_results[:5]:
                print(
                    r.get("id"),
                    r.get("location"),
                    r.get("bhk_type")
                )

            print("==============================\n")

            results_df = pd.DataFrame(raw_results)
            
            matched_full_df = GLOBAL_MASTER_DF[GLOBAL_MASTER_DF["id"].isin(results_df["id"])].copy()
            matched_full_df = matched_full_df.merge(results_df[["id", "search_score"]], on="id", how="left")
            matched_full_df = matched_full_df.rename(columns={"search_score": "cosine_similarity"})

            # Ensure weights logic handles empty state fallbacks
            if sum(synthesized_chat_weights.values()) == 0 and historical_weights:
                synthesized_chat_weights = historical_weights

            # =====================================================================
            # PRODUCTION RUNTIME DEBUG TELEMETRY 
            # =====================================================================
            print("\n" + "="*50)
            print(f"🔍 RUNTIME RANKING TELEMETRY (SOURCE: {extraction_source.upper()})")
            print(f"EXTRACTED FILTERS   : {extracted_filters}")
            print(f"INTENT WEIGHTS RAW  : {synthesized_chat_weights}")
            print(f"SLIDER WEIGHTS RAW  : {current_ui_sliders}")
            print("="*50 + "\n")


            # Execute unified ranker using calculated weights
            ranked_df = apply_hybrid_ranking(
                similar_df=matched_full_df, 
                intent_weights=synthesized_chat_weights, 
                slider_weights=current_ui_sliders, 
                alpha=0.65,
            )

            # Programmatically inject naturalized recommendation reasons
            ranked_df["why_recommended"] = ranked_df.apply(
                lambda row: generate_custom_recommendation_reason(row, preferences), axis=1
            )

            # Record operational tracking history safely inside SQLite store
            st.session_state["last_search_filters"] = extracted_filters
            st.session_state["last_search_weights"] = synthesized_chat_weights
            st.session_state["last_search_preferences"] = preferences

            ranked_df = ranked_df.rename(columns={"hybrid_score": "search_score"})
            ranked_df["amenities_mcp"] = ranked_df.get("amenities_mcp", "")
            
            final_cols = [
                "id",
                "price",
                "bhk_type",
                "location",
                "amenities_mcp",
                "search_score",
                "why_recommended"
            ]

            display_cols = [
                c for c in final_cols
                if c in ranked_df.columns
            ]

            # ---------------------------
            # Pagination
            # ---------------------------

            page = st.session_state.get(
                "search_page",
                0
            )

            page_size = 5

            start_idx = page * page_size
            end_idx = start_idx + page_size

            print(
                f"PAGE={page} "
                f"START={start_idx} "
                f"END={end_idx}"
            )

            final_records = (
                ranked_df[display_cols]
                .iloc[start_idx:end_idx]
                .to_dict(orient="records")
            )
            return {
                "type": "search_results",
                "content": final_records,
                "current_query_state": {
                    "active_filters": extracted_filters,
                    "chat_preference_weights": synthesized_chat_weights,
                    "preferences_telemetry": preferences,
                    "extraction_source": extraction_source
                }
            }
        else:
            return {"type": "text", "content": f"❌ Zero properties matched infrastructure specifications: `{extracted_filters}`."}

    # -----------------------------------------------------------------
    # STEP 5: DEEPSEEK GENERIC CHAT FALLBACK
    # -----------------------------------------------------------------
    staged_context = GLOBAL_MASTER_DF[GLOBAL_MASTER_DF["id"].isin(session_state_tray)].head(3).to_string(index=False) if session_state_tray else "No active properties staged."
    chat_prompt = f"""You are an expert real estate consultant. Answer the inquiry directly.

    ACTIVE CONTEXT ROWS IN USER MEMORY TRAY:
    {staged_context}

    USER REQUEST INPUTS: {user_prompt}
    Provide structured clear insights utilizing Indian Rupee (₹) denominations.
    """
    return {"type": "text", "content": ask_deepseek(chat_prompt)}
    # so it return something like 
    # response = {
    #     "type": "search_results",
    #     "content": [
    #         {"id": "P101", "price": 2.1},
    #         {"id": "P102", "price": 2.4}
    #     ]
    # }







# What does extract_intent_and_preferences() do?
#
# This function understands the user's property search query and converts it
# into structured information that the recommendation engine can use.
#
# It extracts:
# 1. Hard Filters      -> Used to eliminate non-matching properties.
# 2. Soft Preferences  -> Used to understand what the user prioritizes.
# 3. Ranking Weights   -> Numerical weights used for property scoring.
#
#
# Example User Query:
#
# "Show me affordable 3 BHK flats in Thane with gym and swimming pool."
#
#
# -------------------------------------------------------------------------
# Step 1:
# The user's query is sent to the LLM for semantic understanding.
# -------------------------------------------------------------------------
#
# User Query:
#
# "Show me affordable 3 BHK flats in Thane with gym and swimming pool."
# Code responsible:
# ask_deepseek(llm_payload_prompt)
#
# The LLM extracts:
#
# {
#     "filters": {
#         "bhk": "3bhk",
#         "location": "Thane",
#         "amenities": "gym, swimming pool"
#     },
#
#     "preferences": {
#         "price_importance": "high",
#         "amenities_importance": "high",
#         "location_importance": "none",
#         "connectivity_importance": "none",
#         "area_importance": "none"
#     }
# }
#
# -------------------------------------------------------------------------
# Step 2:
# Convert the LLM JSON response into a Python dictionary.
# -------------------------------------------------------------------------
#
# Code responsible:
# parsed_data = json.loads(llm_raw_response)
#
#
# -------------------------------------------------------------------------
# Step 3:
# Extract the hard filters returned by the LLM.
# -------------------------------------------------------------------------
#
# Code responsible:
# filters = parsed_data.get("filters", ...)
#
#
# Example:
#
# {
#     "bhk": "3bhk",
#     "location": "Thane",
#     "amenities": "gym, swimming pool"
# }
#
#
# -------------------------------------------------------------------------
# Step 4:
# Extract the user preference importance levels.
# -------------------------------------------------------------------------
#
# Code responsible:
# preferences = parsed_data.get("preferences", {})
#
#
# Example:
#
# {
#     "price_importance":"high",
#     "amenities_importance":"high",
#     "location_importance":"none",
#     ...
# }
# -------------------------------------------------------------------------
# Step 5:
# Apply backup logic if the LLM misses information.
# --------------------------------------------------------------------------
#
# Code responsible:
#
# bhk_match = re.search(...)
#
# filters["location"] = loc
#
# preferences["amenities_importance"] = "high"
#
#
# Example:
#
# User:
# "Need a flat with gym."
#
# Even if the LLM misses "gym",
# the metadata lookup detects it and sets:
#
# preferences["amenities_importance"] = "high"
#
#
# -------------------------------------------------------------------------
# Step 6:
# Convert textual preference levels into numerical weights.
# -------------------------------------------------------------------------
# Code responsible:
#
# synthesized_weights = {
#     ...
# }
#
# Text Preferences:
#
# {
#     "price_importance": "high",
#     "amenities_importance": "high",
#     "location_importance": "none",
#     "connectivity_importance": "none",
#     "area_importance": "none"
# }
#
#
# become
#
# {
#     "price": 0.75,
#     "amenities": 0.75,
#     "location": 0.0,
#     "connectivity": 0.0,
#     "area": 0.0,
#     "distance": 0.0
# }
#
# These numerical weights are later used by the recommendation engine
# while ranking properties.
#
#
# -------------------------------------------------------------------------
# Step 7:
# Handle follow-up conversations by reusing previous filters and weights.
# -------------------------------------------------------------------------
#
# Code responsible:
#
# if is_followup_query(prompt_lower):
#     ...
#
#
# Example:
#
# Previous query:
# "Show me 2 BHK in Thane."
#
# Follow-up:
# "Only under 90 lakhs."
#
# The previous location and BHK are automatically reused.
#
#
# -------------------------------------------------------------------------
# Step 8:
# Return the final structured output.
# ---------------------------------------------------------------------------
#
# Code responsible:
#
# return {
#     "filters": filters,
#     "preferences": preferences,
#     "weights": synthesized_weights,
#     "source": "llm_unified_parser"
# }
#
#
# {
#     "filters": {
#         "bhk": "3bhk",
#         "location": "Thane",
#         "amenities": "gym, swimming pool"
#     },
#
#     "preferences": {
#         "price_importance": "high",
#         "amenities_importance": "high",
#         "location_importance": "none",
#         "connectivity_importance": "none",
#         "area_importance": "none"
#     },
#
#     "weights": {
#         "price": 0.75,
#         "amenities": 0.75,
#         "location": 0.0,
#         "connectivity": 0.0,
#         "area": 0.0,
#         "distance": 0.0
#     },
#
#     "source": "llm_unified_parser"
# }
#
#
# -------------------------------------------------------------------------
# How each returned value is used
# -------------------------------------------------------------------------
#
# filters
# --------
# Hard constraints used to filter the property dataset.
#
# Example:
#
# Keep only:
# ✓ 3 BHK
# ✓ Located in Thane
# ✓ Having Gym and Swimming Pool
#
#
# preferences
# -----------
# Human-readable importance levels detected from the user's query.
#
# Example:
#
# Price      -> High priority
# Amenities  -> High priority
# Location   -> Normal (not specially prioritized)
#
#
# weights
# -------
# Numerical version of the preferences.
#
# These weights are passed to the property scoring engine so that
# properties matching important user preferences receive higher scores.
#
#
# source
# ------
# Indicates which parser produced the result.
#
# Example:
#
# "llm_unified_parser"
#
# This is useful for debugging or knowing whether the LLM or a fallback
# parser generated the extracted information.
#
#
# Finally, this structured output is passed to the property search and
# recommendation engine, where:
#
# 1. Filters remove irrelevant properties.
# 2. Weights rank the remaining properties according to the user's priorities.
# 3. The highest-scoring properties are returned to the user.


# -------------------------------------------------------------------------
# LAYER 2 : DETERMINISTIC REGEX HEURISTIC FALLBACK
# -------------------------------------------------------------------------
#
# This layer is executed only when the LLM parsing fails
# (for example, invalid JSON, timeout, or any exception).
#
# Instead of using the LLM, this layer uses:
#
# 1. Regular Expressions (Regex)
# 2. Search Metadata (locations & amenities)
# 3. Keyword Dictionaries
#
# to produce exactly the same output format as the LLM.
#
#
# Example User Query:
#
# "Show me an affordable 3 BHK in Thane with gym near metro."
#
#
# -------------------------------------------------------------------------
# Step 1:
# Create an empty filter dictionary.
# -------------------------------------------------------------------------
#
# Code responsible:
#
# extracted_filters = {
#     "bhk": None,
#     "amenities": None,
#     "location": None
# }
#
#
# -------------------------------------------------------------------------
# Step 2:
# Extract the BHK using a regular expression.
# -------------------------------------------------------------------------
#
# Code responsible:
#
# bhk_match = re.search(FILTER_INTENTS["bhk_pattern"], prompt_lower)
#
#
# Example:
#
# User Query:
#
# "Need affordable 3 BHK in Thane"
#
#
# Result:
#
# {
#     "bhk":"3bhk"
# }
#
# If no BHK is found,
# reuse the BHK from the previous conversation.
#
#
# -------------------------------------------------------------------------
# Step 3:
# Extract the location using the metadata dictionary.
# -------------------------------------------------------------------------
#
# Code responsible:
#
# known_locations = CACHED_SEARCH_METADATA.get("location", [])
#
# for loc in known_locations:
#     ...
#
#
# Example Metadata:
#
# [
#     "thane",
#     "andheri",
#     "powai",
#     "goregaon"
# ]
#
#
# User Query:
#
# "Need a flat in Thane"
#
#
# Result:
#
# {
#     "location":"Thane"
# }
#
#
# If no location is found,
# reuse the previous location.
#
#
# -------------------------------------------------------------------------
# Step 4:
# Extract amenities using the metadata dictionary.
# -------------------------------------------------------------------------
#
# Code responsible:
#
# known_amenities = CACHED_SEARCH_METADATA.get("amenities_mcp", [])
#
# for amenity in known_amenities:
#     ...
#
#
# Example Metadata:
#
# [
#     "gym",
#     "swimming pool",
#     "club house"
# ]
#
#
# User Query:
#
# "Need gym and swimming pool"
#
#
# Result:
#
# {
#     "amenities":"gym swimming pool"
# }
#
#
# -------------------------------------------------------------------------
# Step 5:
# Initialize the default ranking weights.
# -------------------------------------------------------------------------
#
# Code responsible:
#
# base_weights = DEFAULT_BLANK_WEIGHTS.copy()
#
#
# Example:
#
# {
#     "price":0,
#     "location":0,
#     "amenities":0,
#     "connectivity":0,
#     "area":0,
#     "distance":0
# }
#
#
# -------------------------------------------------------------------------
# Step 6:
# Detect ranking-related keywords from the user's query.
# -------------------------------------------------------------------------
#
# Code responsible:
#
# for intent_name, keywords in RANKING_WORD_LISTS.items():
#
#
# Example User Query:
#
# "Affordable flat near metro"
#
#
# Keywords Found:
#
# affordable
# metro
#
#
# -------------------------------------------------------------------------
# Step 7:
# Ignore keywords that are negated.
# -------------------------------------------------------------------------
#
# Code responsible:
#
# if any(re.search(neg, preceding_chunk) for neg in NEGATIONS):
#     continue
#
#
# Example:
#
# User Query:
#
# "Not near metro"
#
#
# Since "not" appears before "metro",
# the connectivity intent is ignored.
#
#
# -------------------------------------------------------------------------
# Step 8:
# Increase the strength of important keywords.
# -------------------------------------------------------------------------
#
# Code responsible:
#
# for modifier, multiplier in INTENSITY_MODIFIERS.items():
#
#
# Example:
#
# User Query:
#
# "Very affordable"
#
#
# Instead of:
#
# price strength = 1.0
#
#
# it becomes:
#
# price strength = 1.5
#
#
# -------------------------------------------------------------------------
# Step 9:
# Store every detected ranking intent.
# -------------------------------------------------------------------------
#
# Code responsible:
#
# intent_quality_logs[intent_name] = {
#     ...
# }
#
#
# Example:
#
# {
#     "cheap":{
#         "strength":1.5,
#         "match_quality":0.95
#     }
# }
#
#
# -------------------------------------------------------------------------
# Step 10:
# Convert detected intents into numerical feature weights.
# -------------------------------------------------------------------------
#
# Code responsible:
#
# target_map = RANKING_TARGET_MAPS.get(intent_name,{})
#
# base_weights[feature] += ...
#
#
# Example:
#
# cheap
#     ↓
# price += 1.0
#
#
# metro
#     ↓
# connectivity += 1.0
# distance += 0.8
#
#
# Final Weights:
#
# {
#     "price":1.0,
#     "connectivity":1.0,
#     "distance":0.8
# }
#
#
# -------------------------------------------------------------------------
# Step 11:
# Reconstruct the preferences dictionary so it matches the LLM format.
# -------------------------------------------------------------------------
#
# Code responsible:
#
# synthesized_preferences = {
#     ...
# }
#
#
# Example:
#
# {
#     "price_importance":"high",
#     "connectivity_importance":"high",
#     "amenities_importance":"none"
# }
#
#
# -------------------------------------------------------------------------
# Step 12:
# Return the final output.
# -------------------------------------------------------------------------
#
# Code responsible:
#
# return {
#     "filters": extracted_filters,
#     "preferences": synthesized_preferences,
#     "weights": base_weights,
#     "source": "regex_fallback"
# }
#
#
# Final Output:
#
# {
#     "filters":{
#         "bhk":"3bhk",
#         "location":"Thane",
#         "amenities":"gym"
#     },
#
#     "preferences":{
#         "price_importance":"high",
#         "connectivity_importance":"high",
#         "amenities_importance":"none",
#         "location_importance":"none",
#         "area_importance":"none"
#     },
#
#     "weights":{
#         "price":1.0,
#         "connectivity":1.0,
#         "distance":0.8,
#         "location":0.0,
#         "amenities":0.0,
#         "area":0.0
#     },
#
#     "source":"regex_fallback"
# }
#
#
# -------------------------------------------------------------------------
# How each returned value is used
# -------------------------------------------------------------------------
#
# filters
# --------
# Used to filter the property dataset.
#
# Example:
#
# ✓ 3 BHK
# ✓ Thane
# ✓ Gym
#
#
# preferences
# -----------
# Human-readable importance levels generated using regex rules.
#
#
# weights
# -------
# Numerical weights used by the recommendation engine to rank properties.
#
#
# source
# ------
# Indicates that the output was generated by the regex fallback layer
# instead of the LLM.
#
#
# Finally, this output has exactly the same structure as the LLM output.
# Therefore, the recommendation engine can use it without knowing whether
# the data came from the LLM parser or the regex fallback parser.