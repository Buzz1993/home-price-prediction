# ===============================
# future_agent.py (FINAL FIXED)
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
    if not isinstance(text, str):
        return ""
    return text.lower()


# -----------------------------
# EXTRACT SIGNALS
# -----------------------------
def extract_future_signal(text):
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

    results = []

    for _, row in df.iterrows():

        text = " ".join([
            str(row.get("features_text", "")),
            str(row.get("nearest_text", ""))   
        ])

        future_words, infra_words = extract_future_signal(text)

        label = growth_label(future_words, infra_words)
        reason = growth_reason(future_words, infra_words)
        score = growth_score(label)

        results.append({
            "id": row.get("id"),
            "future_signals": ", ".join(future_words),
            "infra_detected": ", ".join(infra_words),
            "growth_label": label,
            "growth_reason": reason,
            "growth_score": score   # optional but useful
        })

    return results