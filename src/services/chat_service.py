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

# so this new below code is made for the user intent project and we have keep this above earlier code
# as it is below 100% just added new extra function parse_intent_and_execute


# ===============================
# chat_service.py
# ===============================
import re
import json
from pathlib import Path
import pandas as pd
from src.utils.search_engine import RealEstateSearchEngine
from src.mcp.tools.comparison_tools import compare_properties
from src.llm.memory_store import SQLiteMemoryStore

# Initialize structural resources
ROOT_PATH = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT_PATH / "data" / "cleaned" / "final_combined_mcp_data.csv"

search_engine = RealEstateSearchEngine(DATA_FILE)
memory_store = SQLiteMemoryStore()

USER_ID = "default_user"

# Operational tracking filters
CHAT_STOPWORDS = {
    "want", "need", "show", "me", "find", "get", "property", "properties",
    "with", "and", "a", "an", "the", "for", "in", "at", "house", "flat", 
    "apartment", "please", "give", "share", "only", "near", "to", "from", "but", "i"
}

def build_context(
    recs=None,
    comparison_result=None,
    comparison_raw=None,
    last_explanation=None
):
    """
    Build compact property context for LLM.

    Priority:
    1. Input property
    2. Comparison raw (full enriched property data)
    3. Comparison result (summary scores/verdict)
    4. Explanation
    """

    sections = []

    # =====================================
    # INPUT PROPERTY
    # =====================================
    # Extract selected columns and their values from recs["input"] and add them to the sections list for LLM context generation.
    if recs and "input" in recs:

        input_df = recs["input"]

        if not input_df.empty:

            cols = [
                c for c in [
                    "id",
                    "project_name",
                    "location",
                    "price",
                    "area",
                    "bhk_type"
                ]
                if c in input_df.columns
            ]

            sections.append(
                "INPUT PROPERTY:\n"
                + input_df[cols].to_string(index=False)
            )

    # =====================================
    # ENRICHED PROPERTY DATA
    # =====================================
    if (
        comparison_raw is not None
        and not comparison_raw.empty
    ):

        important_cols = [

            # identity
            "id",
            "project_name",
            "location",

            # recommendation
            "why_recommended",
            "hybrid_score",

            # valuation
            "analysis_msg",

            # risk
            "risk_label",
            "risk_score",

            # growth
            "growth_label",
            "growth_reason",

            # rental
            "monthly_rent_estimate",
            "rental_yield_percent",
            "investment_rating",

            # negotiation
            "negotiation_power",
            "suggested_discount_percent",
            "target_price",
            "price_position",

            # development
            "dev_summary"
        ]

        cols = [
            c
            for c in important_cols
            if c in comparison_raw.columns
        ]

        sections.append(
            "PROPERTY DATA:\n"
            + comparison_raw[cols].to_string(index=False)
        )

    # =====================================
    # COMPARISON SUMMARY
    # =====================================
    if (
        comparison_result is not None
        and not comparison_result.empty
    ):

        cols = [
            c for c in [
                "id",
                "overall_score",
                "verdict",
                "comparison_reason"
            ]
            if c in comparison_result.columns
        ]

        sections.append(
            "COMPARISON SUMMARY:\n"
            + comparison_result[cols].to_string(index=False)
        )

    # =====================================
    # EXPLANATION
    # =====================================
    if last_explanation:

        sections.append(
            "COMPARISON INSIGHTS:\n"
            + str(last_explanation)
        )

    if not sections:
        return "No property data available."

    return "\n\n".join(sections)


def extract_search_tokens(text: str):
    """Isolates pristine structural search tokens from human entries."""
    raw_words = text.lower().strip().split()
    # Heavily check and drop anything containing 'bhk' or matching noise tokens
    clean_tokens = [w for w in raw_words if w not in CHAT_STOPWORDS and "bhk" not in w]
    return " ".join(clean_tokens) if clean_tokens else None


def parse_intent_and_execute(user_prompt: str, session_state_tray: list) -> dict:
    """
    Evaluates chat intent and syncs active requirements directly with 
    your persistent SQLite database layers to handle conversational memory.
    """
    import streamlit as st
    prompt_lower = user_prompt.lower().strip()
    words = prompt_lower.split()
    
    # -----------------------------------------------------------------
    # STEP 1: Comparison Analysis Trigger
    # -----------------------------------------------------------------
    comparison_keywords = [
        "compare",
        "ranking",
        "rank"
    ]

    if any(k in prompt_lower for k in comparison_keywords):
        if len(session_state_tray) < 2:
            return {
                "type": "text",
                "content": "⚠️ I need at least 2 properties in your tray to run an investment comparison. Try searching and adding properties first!"
            }
        
        print("🔄 Routing context directly to Comparison Node Module...")
        comparison_raw_json = compare_properties(session_state_tray)
        comparison_data = json.loads(comparison_raw_json)
        
        return {
            "type": "comparison",
            "content": comparison_data
        }

    # -----------------------------------------------------------------
    # STEP 2: Structural Parameter Isolation
    # -----------------------------------------------------------------
    extracted_criteria = {"bhk": None, "amenities": None, "location": None}
    
    # Isolate BHK constraints cleanly
    match = re.search(
        r'(\d+)\s*bhk',
        prompt_lower
    )

    if match:
        extracted_criteria["bhk"] = f"{match.group(1)}bhk"
    else:
        for idx, word in enumerate(words):
            if word == "bhk" and idx > 0 and words[idx-1].isdigit():
                extracted_criteria["bhk"] = f"{words[idx-1]}bhk"
                break

    # Positional separator parsing (Checks for 'near' or 'from')
    separator_word = None
    if "near" in words:
        separator_word = "near"
    elif "from" in words:
        separator_word = "from"

    if separator_word:
        sep_idx = words.index(separator_word)
        extracted_criteria["location"] = extract_search_tokens(" ".join(words[sep_idx + 1:]))
        extracted_criteria["amenities"] = extract_search_tokens(" ".join(words[:sep_idx]))
    else:
        # If no strict separator, clean the whole string for attributes
        extracted_criteria["amenities"] = extract_search_tokens(user_prompt)
        extracted_criteria["location"] = extract_search_tokens(user_prompt)

    # -----------------------------------------------------------------
    # STEP 3: Persistent SQLite Long-Term Memory Lookup & Merge
    # -----------------------------------------------------------------
    saved_memories = memory_store.get_memories(USER_ID)
    historical_state = {}
    for memory_string in saved_memories:
        try:
            parsed_block = json.loads(memory_string)
            if isinstance(parsed_block, dict):
                historical_state.update(parsed_block)
        except json.JSONDecodeError:
            continue

    # --- THE CRITICAL BUG FIX LAYER ---
    # Only fall back to old memory if the user did NOT type a fresh location in their current prompt!
    if not extracted_criteria["location"]:
        # If no fresh location token was provided, pass forward old location memory safely
        if historical_state.get("location"):
            extracted_criteria["location"] = historical_state["location"]
    else:
        pass

    if not extracted_criteria["bhk"] and historical_state.get("bhk"):
        extracted_criteria["bhk"] = historical_state["bhk"]

    # -----------------------------------------------------------------
    # STEP 4: Persist the Fresh Combined Query back to SQLite
    # -----------------------------------------------------------------
    active_preferences = {k: v for k, v in extracted_criteria.items() if v}
    if active_preferences:
        memory_store.add_memory(USER_ID, json.dumps(active_preferences))

    print(f"🧠 SQLite Merged System State Processing: {extracted_criteria}")
    
    # -----------------------------------------------------------------
    # STEP 5: Fire BM25 Matrix Matcher + Pandas HARD FILTERING
    # -----------------------------------------------------------------
    determined_min_matches = 1 if (not extracted_criteria["amenities"] or not extracted_criteria["location"]) else 2
    results_df = search_engine.query(extracted_criteria, min_matches=determined_min_matches)
    
    if results_df.empty:
        return {
            "type": "text",
            "content": f"❌ Zero properties matched your search parameters: `{extracted_criteria}`."
        }
    
    # --- THE CRITICAL HARD FILTERING LAYER ---
    # If a specific BHK value exists in our search state, forcefully prune the output frame
    if extracted_criteria["bhk"]:
        target_bhk = extracted_criteria["bhk"].strip().lower() # e.g., "1 bhk" or "2 bhk"
        
        # Build an exact string matching mask against your df['bhk_type'] column
        if "bhk_type" in results_df.columns:
            # We standardize comparisons to handle potential spacing variations (e.g., "2bhk" vs "2 bhk")
            filtered_df = results_df[
                results_df["bhk_type"].str.lower().str.replace(" ", "") == target_bhk.replace(" ", "")
            ]
            
            # Fallback: If strict filtering leaves us with a completely empty frame, 
            # gracefully return original BM25 results but print a terminal notice.
            if not filtered_df.empty:
                results_df = filtered_df
            else:
                print(f"⚠️ Warning: Strict filter for '{target_bhk}' returned zero rows. Falling back to fuzzy matches.")

    return {
        "type": "search_results",
        "content": results_df.head(5).to_dict(orient="records"),
        "current_query_state": extracted_criteria
    }