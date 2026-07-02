# ===============================
# explanation_agent.py
# ===============================

def generate_comparison_explanation(raw_df, compare_df, llm_func):
    """
    Generate AI-based(using deepseek model) comparison explanation
    for selected properties using property,
    scoring, and development data.

    Returns:
        str -> Final AI-generated comparison explanation.
    """
    print("✅ explanation_agent executed")

    # print("===============================")
    # print("raw_df", raw_df.columns.tolist())
    # print("compare_df", compare_df.columns.tolist())
    # print("===============================")

    # 🔹 Structured data (very important)
    context = {
        "raw_data": raw_df[[
            "id",
            "project_name",
            "location",

            "price",

            "analysis_msg",

            "price_position",
            "negotiation_power",
            "negotiation_score",

            "risk_label",

            "growth_label",
            "growth_reason",

            "dev_summary",

            "locality_rating",

            "monthly_rent_estimate",
            "rental_yield_percent",
            "investment_rating",
            "demand_level",
            "rental_strategy"
        ]].to_dict(orient="records"),

        "scoring": compare_df.to_dict(orient="records"),
        "development_data": raw_df[["id", "dev_summary"]].to_dict(orient="records")
    }

    # 🔹 Prompt (your improved version)
    prompt = f"""
    You are a real estate expert.

    You are given:

    1. RAW PROPERTY DATA including:
        - project and location details
        - price value insight (analysis_msg, price_position)
        - negotiation strength (negotiation_power)
        - risk profile (risk_label)
        - growth potential (growth_label, growth_reason, dev_summary)
        - locality quality (locality_rating)
        - rental performance (monthly rent, rental yield, investment rating)
    2. COMPARISON DATA including overall score, verdict, and comparison reasoning
    3. DEVELOPMENT DATA (latest infrastructure insights if available)

    {context}

    Instructions:
    - Identify the BEST property
    - Use REAL insights from RAW data:
    - location and locality quality
    - infrastructure / future growth
    - price value (not just number, but value)
    - risk factors
    - Use fields like analysis_msg, growth_reason if available
    - DO NOT mention or rely on scores directly
    - DO NOT repeat raw numbers blindly
    - Be practical and human-like
    - Use development_data if available for growth comparison
    - Prefer real-time developments over static growth_reason
    - Use dev_summary whenever available
    - Mention specific upcoming infrastructure projects if present
    - Prefer dev_summary over generic growth_reason when both exist
    - MUST compare rental yield and income potential between properties
    - Mention which property gives better rental return
    - Use rental data as a key decision factor
    - If one property has significantly better rental yield, prioritize it as investment
    - You will receive MULTIPLE properties (2 to 10)
    - You MUST analyze ALL properties, not just 2
    - Compare ALL of them before giving final verdict
    - IMPORTANT: All prices are in Indian Rupees (₹)
    - NEVER use dollars ($)
    - Format like: ₹2.35 Cr
    - Consider negotiation strength, negotiation score and pricing position
    - Use demand_level and rental_strategy when discussing rental attractiveness
    - Mention if a property looks overpriced or undervalued
    - DO NOT choose winner solely based on overall_score
    - Cross-check price value, risk, growth, rental yield and negotiation opportunity

    Output STRICTLY in this format:

    🏆 Best Property (among all selected): <id>

    Why it's better:
    - 2–3 simple bullet points using REAL insights

    Comparison:

    For EACH property:

    1. Price
    - <id>: explain value (cheap / overpriced / justified)

    2. Risk
    - <id>: explain real risks

    3. Growth
    - <id>: explain growth potential

    4. Rental / Investment Potential
    - <project_name or id>: explain rental yield and investment quality

    Verdict:
    - Winner: <id> (Best Value / Balanced / Risky)
    - Why: 1-line practical summary

    Avoid:
    - <id> (if applicable)
    - Reason: short explanation

    If both properties are similar:
    - Clearly say they are comparable and explain the difference briefly
    """

    # 🔹 LLM call
    full_response = ""
    for chunk in llm_func(prompt):
        full_response += chunk

    return full_response