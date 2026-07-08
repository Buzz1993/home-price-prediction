# ===============================
# risk_agent.py
# ===============================

import re

# -----------------------------
# KEYWORDS
# -----------------------------
RISK_KEYWORDS = {

    # 🚗 Traffic & Mobility
    "traffic": [
        "traffic", "congestion", "traffic jam", "heavy traffic",
        "peak hours", "long commute", "delays",
        "narrow roads", "road blockage"
    ],

    # 🅿️ Parking
    "parking": [
        "parking issue", "limited parking", "no parking",
        "visitor parking", "parking congestion"
    ],

    # 🛣️ Roads & Infra
    "infrastructure": [
        "potholes", "poor road condition", "bad roads",
        "waterlogging", "drainage issue", "bumpy roads"
    ],

    # 🚆 Connectivity
    "connectivity": [
        "no metro", "far railway station", "poor public transport",
        "limited transport", "difficult access", "remote location"
    ],

    # 💧 Water
    "water": [
        "water shortage", "irregular water supply",
        "poor water quality", "water scarcity", "tanker water"
    ],

    # ⚡ Electricity
    "power": [
        "power cut", "power outage", "electricity issues",
        "no power backup", "unreliable electricity"
    ],

    # 🌫️ Pollution
    "pollution": [
        "air pollution", "noise pollution", "dust",
        "pollution", "construction noise", "vehicle pollution"
    ],

    # 🏗️ Construction
    "construction": [
        "ongoing construction", "construction noise",
        "delayed possession", "under construction"
    ],

    # 🏢 Maintenance
    "maintenance": [
        "poor maintenance", "leakage", "seepage",
        "plumbing issue", "electrical fault",
        "lift problem", "poor construction quality"
    ],

    # 👮 Safety
    "safety": [
        "theft", "crime", "unsafe",
        "poor police presence", "assault", "unsafe at night"
    ],

    # 🧍 Overcrowding
    "overcrowding": [
        "overcrowded", "densely populated",
        "high population density"
    ],

    # 🧹 Cleanliness
    "cleanliness": [
        "garbage", "unhygienic",
        "poor cleanliness", "waste management issue"
    ],

    # 🌊 Flooding
    "flooding": [
        "flooding", "waterlogging", "drainage failure"
    ],

    # 💸 Financial
    "financial": [
        "high rent", "expensive", "overpriced",
        "high maintenance charges", "affordability issue"
    ],

    # 🏫 Amenities
    "amenities": [
        "no hospital", "no school",
        "lack of amenities", "limited facilities"
    ],

    # 🐕 Lifestyle
    "lifestyle": [
        "street dogs", "vendors blocking roads",
        "food delivery not available"
    ]
}

# -----------------------------
# WEIGHTS
# -----------------------------
RISK_WEIGHTS = {
    "traffic": 2,
    "parking": 2,
    "pollution": 2,
    "water": 3,
    "power": 2,
    "safety": 3,
    "maintenance": 2,
    "construction": 1,
    "overcrowding": 2,
    "cleanliness": 1,
    "connectivity": 2,     
    "infrastructure": 2,   
    "flooding": 3,         
    "financial": 2,
    "amenities": 1,
    "lifestyle": 1
}

# -----------------------------
# TEXT CLEAN
# -----------------------------
def clean_text(text):
    """
    Clean and normalize input text.
    """
    if not isinstance(text, str):
        return ""
    return text.lower()

# -----------------------------
# EXTRACT RISKS
# -----------------------------
def extract_risks(text):
    """
    Extract risk categories from text
    using keyword matching.
    """
    text = clean_text(text)
    found = set()

    for category, keywords in RISK_KEYWORDS.items():
        for kw in keywords:

            # 1. exact phrase match
            if kw in text:
                found.add(category)
                break

            # Loose keyword matching:Check if all words from keyword exist somewhere in the text.
            # Example: "traffic jam" 
            # "heavy traffic because of road jam"
            words = kw.split()
            if all(w in text for w in words):
                found.add(category)
                break

    return list(found)

# -----------------------------
# SCORE
# -----------------------------
def risk_score(risks):
    """
    Calculate total risk score
    based on detected risks.
    """
    return sum(RISK_WEIGHTS.get(r, 1) for r in risks) # Sum weights of all detected risks 
                                                      # If risk category not found in RISK_WEIGHTS,
                                                      # use default weight = 1.

# -----------------------------
# LABEL
# -----------------------------
def risk_label(score):
    """
    Convert numerical risk score
    into Low, Medium, or High risk label.
    """
    if score >= 6:
        return "🔴 High Risk"
    elif score >= 3:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"

# -----------------------------
# MAIN AGENT
# -----------------------------
def run_risk_agent(df):
    """
    Run risk analysis on all properties
    and return results in list format.
    """
    print("🟨 risk_agent executed")
    results = []

    for _, row in df.iterrows():

        #Get value from "needs_improvement" column for current dataframe row.
        text = str(row.get("needs_improvement", "")) #needs_improvement is coilumn name already have in dataframe created while data cleaning

        risks = extract_risks(text)  # Extract detected risk categories from text - Example: ["traffic", "parking"]
        score = risk_score(risks) # Calculate total numerical risk score - Example: traffic(2) + parking(2) = 4
        label = risk_label(score) # Convert numerical score into risk label (Low Risk, Medium Risk, High Risk) - Example: 4 → 🟡 Medium Risk

        results.append({
            "id": row.get("id"),
            "risk_categories": ", ".join(risks),
            "risk_score": score,
            "risk_label": label
        }) #"risk_categories" (comma separated risk categories), "risk_score" (numerical score), "risk_label" (Low, Medium, High)

    return results