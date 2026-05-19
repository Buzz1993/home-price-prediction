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
    "connectivity": 2,     # 👈 added
    "infrastructure": 2,   # 👈 added
    "flooding": 3,         # 👈 high impact
    "financial": 2,
    "amenities": 1,
    "lifestyle": 1
}

# -----------------------------
# TEXT CLEAN
# -----------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.lower()

# -----------------------------
# EXTRACT RISKS
# -----------------------------
def extract_risks(text):
    text = clean_text(text)
    found = set()

    for category, keywords in RISK_KEYWORDS.items():
        for kw in keywords:

            # 1. exact phrase match
            if kw in text:
                found.add(category)
                break

            # 2. loose match (handles variations)
            words = kw.split()
            if all(w in text for w in words):
                found.add(category)
                break

    return list(found)

# -----------------------------
# SCORE
# -----------------------------
def risk_score(risks):
    return sum(RISK_WEIGHTS.get(r, 1) for r in risks)

# -----------------------------
# LABEL
# -----------------------------
def risk_label(score):
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

    results = []

    for _, row in df.iterrows():

        text = str(row.get("needs_improvement", ""))

        risks = extract_risks(text)
        score = risk_score(risks)
        label = risk_label(score)

        results.append({
            "id": row.get("id"),
            "risk_categories": ", ".join(risks),
            "risk_score": score,
            "risk_label": label
        })

    return results