# ===============================
# hybrid_recommender.py 
# ===============================

import pandas as pd

def normalize(series):
    """Normalize values between 0 and 1 for scale unification."""
    #print("☑️ normalize executed")
    if series.max() == series.min():
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.min()) / (series.max() - series.min())



# -----------------------------
# INTENT WEIGHTS
# -----------------------------
def get_dynamic_weights(intent):
    """
    Generate recommendation weights
    based on user preferences like:
    luxury, low budget, spacious, etc.

    Returns normalized weights.
    """
    print("☑️ get_dynamic_weights executed")

    weights = {
        "price": 0.30,
        "area": 0.20,
        "amenities": 0.15,
        "location": 0.15,
        "connectivity": 0.10,
        "distance": 0.10
    }

    if not intent:
        return weights

    preferences = intent.get("preferences", [])

    if "low budget" in preferences: #if user has "low budget" preference then we increase the weight for price and decrease the weight for area because for low budget users price is more important than area
        weights["price"] += 0.15
        weights["area"] -= 0.05

    if "luxury" in preferences:
        weights["amenities"] += 0.15
        weights["area"] += 0.10
        weights["price"] -= 0.10

    if "location" in preferences:
        weights["location"] += 0.15
        weights["distance"] += 0.10
        weights["price"] -= 0.05

    if "spacious" in preferences:
        weights["area"] += 0.20
        weights["price"] -= 0.05

    if "investment" in preferences:
        weights["price"] += 0.10
        weights["location"] += 0.10

    total = sum(weights.values())
    for k in weights:
        weights[k] /= total

    return weights



def sanitize_weights(weights, baseline_keys):
    """
    Validate and normalize a weight dictionary.

    - Ensures all expected keys exist.
    - Converts values to float.
    - Replaces missing or invalid values with 0.0.
    """
    #print("☑️ sanitize_weights executed")
    if not isinstance(weights, dict):
        return {k: 0.0 for k in baseline_keys}

    cleaned = {}

    for k in baseline_keys:
        value = weights.get(k, 0.0)

        try:    
            cleaned[k] = float(value)
        except (TypeError, ValueError):
            cleaned[k] = 0.0

    return cleaned


def combine_weights(intent_weights, slider_weights):
    """
    Combine preferences from 2 sources:
    1. UI sliders (60%) -> what user manually selected
    2. Chat intent (40%) -> what user asked in conversation
    The final weights are normalized to sum to 1.0.
    Normalize and merge all active sources into one final weight dictionary.
    If no preferences are available, use default system weights.

    Returns:
    dict: Final ranking weights.
    """

    print("☑️ combine_weights executed")
    fallback_baseline = {
        "price": 0.30, "area": 0.20, "amenities": 0.15,
        "location": 0.15, "connectivity": 0.10, "distance": 0.10
    }

    sanitized_intent = sanitize_weights(intent_weights, fallback_baseline.keys())
    sanitized_slider = sanitize_weights(slider_weights, fallback_baseline.keys())
    

    sources = []
    
    # 1. UI Sliders: Contribute whenever slider weights are available
    if sum(sanitized_slider.values()) > 0:
        ui_total = sum(sanitized_slider.values())
        ui_norm = {k: v / ui_total for k, v in sanitized_slider.items()}
        sources.append((ui_norm, 0.60)) # 0.60 means 60% weight for sliders in final weight combination
        
    # 2. Chat Intents: Only contribute if conversational keywords were extracted
    if sum(sanitized_intent.values()) > 0:
        chat_total = sum(sanitized_intent.values())
        chat_norm = {k: v / chat_total for k, v in sanitized_intent.items()}
        sources.append((chat_norm, 0.40)) # 0.40 means 40% weight for chat intent in final weight combination
        
        
    # Fallback Mechanism: If no active layers matched, return calibrated uniform system defaults
    if not sources: # if sources = [] then return default {"price": 0.30, "area": 0.20, "amenities": 0.15,"location": 0.15, "connectivity": 0.10, "distance": 0.10}
        return fallback_baseline

    # Normalize source blending ratios to account for missing layers dynamically
    total_source_share = sum(share for _, share in sources)
    normalized_sources = [(w, share / total_source_share) for w, share in sources]

    # Synthesize unified vector spaces
    blended_weights = {k: 0.0 for k in fallback_baseline.keys()} # Create empty final weights dictionary with all keys initialized to 0.0
    for weights, share in normalized_sources:
        for key in blended_weights.keys():
            blended_weights[key] += weights.get(key, 0.0) * share

    # Force strict calibration scaling (Sum up to exactly 1.0)
    total_magnitude = sum(blended_weights.values())
    if total_magnitude > 0:
        return {k: v / total_magnitude for k, v in blended_weights.items()}
    return fallback_baseline


def explain(row):
    """Generate short analytical explanation snippet for property recommendations."""

    #print("☑️ explain executed")
    reasons = []
    if row["price_score"] > 0.2: reasons.append("good price")
    if row["location_score"] > 0.2: reasons.append("great location")
    if row["area_score"] > 0.2: reasons.append("spacious")
    if row["amenities_score"] > 0.15: reasons.append("good amenities")
    return ", ".join(reasons)


def compute_weighted_score(df, weights):
    """
    Calculate score for each property based on:
    price, area, amenities, location, connectivity, and distance.
    Apply user preference weights to each feature.
    Combine all feature scores into one final weighted_score.
    Adds: price_score, area_score, amenities_score, location_score, connectivity_score, distance_score, weighted_score, and why_recommended columns.
    Generate a short explanation for why the property was recommended.
    """
    print("☑️ compute_weighted_score executed")
    temp = df.copy()

    price_norm = normalize(temp["price"])
    area_norm = normalize(temp["area"])
    amenities_norm = normalize(temp["amenities_count"])
    location_norm = normalize(temp["locality_rating"])
    connectivity_norm = normalize(temp["commuting_rating"])
    distance_norm = normalize(temp["distance_to_center_km"])

    # Score assignment (Inverting lower-is-better metrics like price and distance)
    temp["price_score"] = (1 - price_norm) * weights.get("price", 0) # Cheaper property gets a higher score.
    temp["area_score"] = area_norm * weights.get("area", 0) # Larger area gets a higher score.
    temp["amenities_score"] = amenities_norm * weights.get("amenities", 0) # More amenities get a higher score.
    temp["location_score"] = location_norm * weights.get("location", 0) # Better locality rating gets a higher score.
    temp["connectivity_score"] = connectivity_norm * weights.get("connectivity", 0) # Better commuting rating gets a higher score.
    temp["distance_score"] = (1 - distance_norm) * weights.get("distance", 0) # Closer to city center gets a higher score.

    temp["weighted_score"] = (
        temp["price_score"] + temp["area_score"] + temp["amenities_score"] +
        temp["location_score"] + temp["connectivity_score"] + temp["distance_score"]
    )

    temp["why_recommended"] = temp.apply(explain, axis=1)
    return temp


def apply_hybrid_ranking(similar_df, intent=None, intent_weights=None, slider_weights=None, alpha=0.65):
    """
    The Single Source of Truth Ranking Engine.
    Blends structural vector similarity with normalized business rule filters.
    """
    print("☑️ apply_hybrid_ranking executed")
    if intent_weights is None: 
        intent_weights = get_dynamic_weights(intent)

    resolved_weights = combine_weights(
        intent_weights=intent_weights, 
        slider_weights=slider_weights
    )
    temp = compute_weighted_score(similar_df, resolved_weights)

    #for recommendation flow cosine_similarity is not 1 it is diff for each property, but for the mcp flow cosine cosine_similarity = 1.0 for all properties
    temp["hybrid_score"] = (
        alpha * temp["cosine_similarity"] +
        (1 - alpha) * temp["weighted_score"]
    )

    return temp.sort_values("hybrid_score", ascending=False) # sort the recommended properties based on hybrid score in descending order so that the property with 
                                                             # highest hybrid score comes at the top of the recommendation list