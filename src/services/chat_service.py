# ===============================
# chat_service.py 
# ===============================

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

def parse_intent_and_execute(
    user_prompt: str,
    session_state_tray: list[str] | None = None,
    current_ui_sliders: dict | None = None,
    session_state: dict | None = None,
):
    """
    Main EstateMind Copilot chat workflow.

    Supports both Streamlit and FastAPI.

    Streamlit:
        Automatically uses `st.session_state` to maintain
        search history, pagination, filters, and preferences.

    FastAPI:
        Accepts a standard `session_state` dictionary,
        allowing future integration with Redis, database,
        JWT, or other backend session managers.

    Returns a structured response for search, comparison,
    analysis, recommendations, or general chat.
    """

    # -------------------------------------------------
    # SESSION STATE COMPATIBILITY
    # -------------------------------------------------

    if session_state is None:
        try:
            import streamlit as st
            session_state = st.session_state
        except Exception:
            session_state = {}

    # -------------------------------------------------
    # DEFAULT VALUES
    # -------------------------------------------------

    if session_state_tray is None:
        session_state_tray = []

    if current_ui_sliders is None:
        current_ui_sliders = {}


    print("☑️ parse_intent_and_execute executed")
    prompt_lower = user_prompt.lower().strip()

    print("\n===== USER QUERY =====")
    print(user_prompt)
    print("======================\n")

    # If the query is a follow-up (e.g. "show more", "another"), reuse previous search filters/weights
    # and move to the next page. Otherwise, start a fresh search from page 1 with empty history.
    is_followup = is_followup_query(prompt_lower)

    if is_followup:
        historical_filters = session_state.get(
            "last_search_filters",
            {}
        )

        historical_weights = session_state.get(
            "last_search_weights",
            {}
        )

        # Move to next page
        session_state["search_page"] = (
            session_state.get("search_page", 0) + 1
        )

    else:
        historical_filters = {}
        historical_weights = {}

        # New search starts from first page
        session_state["search_page"] = 0

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
            session_state["last_search_filters"] = extracted_filters
            session_state["last_search_weights"] = synthesized_chat_weights
            session_state["last_search_preferences"] = preferences

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

            page = session_state.get(
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
