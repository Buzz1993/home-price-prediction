# ===============================
# development_utils.py
# ===============================

from ddgs import DDGS
from src.llm.deepseek_client import ask_deepseek
import re

# -----------------------------
# CACHE (IMPORTANT)
# -----------------------------
dev_cache = {}

# -----------------------------
# FILTER RESULTS (REDUCE NOISE)
# -----------------------------
def filter_results(results):
    """
    Keep only useful development-related
    search results and remove noisy results.
    """

    cleaned = []

    # -----------------------------
    # PATTERNS (VERY IMPORTANT)
    # -----------------------------
    patterns = [
        # 🚀 HARD INFRA
        r"\bmetro\b", r"metro\s?station", r"metro[-\s]?line",
        r"\brail\b", r"railway", r"rail[-\s]?corridor",
        r"\bairport\b",
        r"\broad\b", r"\bhighway\b", r"expressway",
        r"\bbridge\b", r"bridges",
        r"trans[-\s]?harbour",
        r"\bcorridor\b",

        # 🏢 COMMERCIAL
        r"it\s?park", r"business\s?park", r"corporate\s?hub",

        # 🏙️ SOFT INFRA
        r"\bschool\b", r"\bcollege\b", r"\bhospital\b",
        r"\bmall\b", r"shopping\s?center",
        r"\bpark\b", r"garden", r"playground"
    ]

    for r in results:
        text = (r.get("title", "") + " " + r.get("body", "")).lower()

        # check ANY pattern match
        if any(re.search(p, text) for p in patterns):
            cleaned.append(text)

    return cleaned


# -----------------------------
# CLEAN SUMMARY (REMOVE JUNK)
# -----------------------------
def clean_summary(summary):
    """
    Clean LLM summary output and
    remove vague/generic responses.
    """

    text = summary.lower()

    bad_phrases = [
        "proximity to",
        "located near",
        "known for",
        "has access to"
    ]

    if any(p in text for p in bad_phrases):
        return "No clear new developments (Confidence: Low)"

    return summary.strip()


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def get_development_summary(location, city):
    """
    Fetch development-related news/results
    using DuckDuckGo search and generate
    a short development summary for a location.
    """

    print("🟨 get_development_summary executed")

    key = f"{location}_{city}".lower()

    # -----------------------------
    # CACHE HIT
    # -----------------------------
    if key in dev_cache:
        return dev_cache[key]

    # -----------------------------
    # SEARCH QUERY (IMPROVED)
    # -----------------------------
    query = f"{location} {city} metro OR road OR infrastructure OR school OR hospital OR mall development 2024 2025"

    text_data = []

    try:
        with DDGS() as ddgs: #DDGS is a Python wrapper for DuckDuckGo's search engine, allowing us to perform web searches and retrieve results programmatically.
            results = ddgs.text(query, max_results=5)

            # filter noisy results
            filtered = filter_results(results)

            text_data = filtered[:3]  # limit noise

    except:
        dev_cache[key] = "No clear new developments (Confidence: Low)"
        return dev_cache[key]

    # -----------------------------
    # NO DATA CASE
    # -----------------------------
    combined = " ".join(text_data)

    if not combined.strip():
        dev_cache[key] = "No clear new developments (Confidence: Low)"
        return dev_cache[key]

    # -----------------------------
    # LLM PROMPT (STRICT)
    # -----------------------------
    prompt = f"""
    You are a strict real estate analyst.

    Analyze developments for: {location}

    {combined}

    Rules:
    - Identify ONLY upcoming or under-construction developments
    - Separate:
        * Growth drivers (metro, road, rail, commercial hubs)
        * Livability improvements (schools, malls, hospitals, parks)
    - DO NOT include old/existing amenities
    - DO NOT guess
    - If unclear → say "No clear new developments"

    Output:
    - Max 12 words
    - Single line
    - Mention key development types only
    """

    summary = ask_deepseek(prompt)

    # -----------------------------
    # CLEAN OUTPUT
    # -----------------------------
    summary = clean_summary(summary)

    # -----------------------------
    # CACHE SAVE
    # -----------------------------
    dev_cache[key] = summary

    return summary