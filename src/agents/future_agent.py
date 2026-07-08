# ===============================
# future_agent.py
# ===============================

# -----------------------------
# FUTURE WORDS (WHEN)
# -----------------------------
FUTURE_WORDS = [
    "upcoming",
    "planned",
    "proposed",
    "under construction",
    "new project",
    "developing",
    "in pipeline",
    "expected",
    "announced"
]

# -----------------------------
# INFRA KEYWORDS (WHAT)
# -----------------------------
INFRA_KEYWORDS = {
    "metro": ["metro"],
    "airport": ["airport"],
    "road": [
        "expressway", "highway",
        "coastal road", "sea link",
        "trans harbour link"
    ],
    "rail": ["railway", "corridor"],
    "commercial": [
        "business park", "it park",
        "tech park", "corporate park",
        "commercial hub"
    ],
    "development": [
        "township", "smart city",
        "infrastructure development",
        "area development"
    ]
}

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    """
    Clean and normalize input text.

    Returns:
        str
    """
    if not isinstance(text, str):
        return ""
    return text.lower()


# -----------------------------
# EXTRACT SIGNALS
# -----------------------------
def extract_future_signal(text):
    """
    Extract future growth keywords
    and infrastructure categories
    from text.

    Returns:
        tuple[list, list]

    Example:
        (
            ["upcoming", "planned"],
            ["metro", "road"]
        )
    """
    text = clean_text(text)

    found_future = set()
    found_infra = set()

    for word in FUTURE_WORDS:
        if word in text:
            found_future.add(word)

    for category, keywords in INFRA_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                found_infra.add(category)
                break

    return list(found_future), list(found_infra)


# -----------------------------
# LABEL
# -----------------------------
def growth_label(future_words, infra_words):
    """
    Generate growth label based on
    detected future and infrastructure signals.

    Returns:
        str

    Example:
        "🚀 High Growth"
    """
    if future_words and infra_words:
        return "🚀 High Growth"
    elif infra_words:
        return "📍 Mature Area"
    else:
        return "➖ No Growth Signal"


# -----------------------------
# REASON
# -----------------------------
def growth_reason(future_words, infra_words):
    """
    Generate explanation for growth label.

    Returns:
        str
    """
    if future_words and infra_words:
        return f"Growth expected due to {', '.join(infra_words)} ({', '.join(future_words)})"
    elif infra_words:
        return f"Established area with existing {', '.join(infra_words)}"
    else:
        return "No major future infrastructure signals"


# -----------------------------
# SCORE (optional)
# -----------------------------
def growth_score(label):
    """
    Convert growth label into
    numerical growth score.

    Returns:
        int

    Example:
        3
    """
    if label == "🚀 High Growth":
        return 3
    elif label == "📍 Mature Area":
        return 1
    else:
        return 0


# -----------------------------
# MAIN AGENT
# -----------------------------
def run_future_agent(df):
    """
    Run future growth analysis on all properties
    and return results in list format.

    Returns:
        list[dict]
    """
    print("🟨 future_agent executed")

    results = []

    for _, row in df.iterrows():

        # Combine text from "features_text" and "nearest_text" columns into one single text string for future growth analysis
        text = " ".join([                     
            str(row.get("features_text", "")),
            str(row.get("nearest_text", ""))   
        ])

        # Example:
        # future_words = ["upcoming", "planned"]
        # infra_words = ["metro", "road"]
        future_words, infra_words = extract_future_signal(text)

        label = growth_label(future_words, infra_words) # Generate growth category label Example: "🚀 High Growth" ,"📍 Mature Area" and "➖ No Growth Signal"
        reason = growth_reason(future_words, infra_words) # Example: "Growth expected due to metro, road (upcoming, planned)"
        score = growth_score(label) #score like 3, 1 or 0

        results.append({
            "id": row.get("id"),
            "future_signals": ", ".join(future_words),
            "infra_detected": ", ".join(infra_words),
            "growth_label": label,
            "growth_reason": reason,
            "growth_score": score   # optional but useful
        }) #"future_signals" (comma separated future words found), "infra_detected" (comma separated infrastructure categories detected), -
           #"growth_label" (High Growth, Mature Area, No Growth Signal), "growth_reason" (text explanation of growth label), "growth_score" (numerical score for growth label) in results

    return results