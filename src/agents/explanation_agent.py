#explanation_agent.py

def generate_comparison_explanation(raw_df, score_df, llm_func):

    # 🔹 Structured data (very important)
    context = {
        "raw_data": raw_df[[
            "id",
            "price",
            "analysis_msg",
            "risk_label",
            "growth_label",
            "growth_reason",
            "monthly_rent_estimate",
            "rental_yield_percent",
            "investment_rating"
        ]].to_dict(orient="records"),

        "scoring": score_df.to_dict(orient="records"),
        "development_data": raw_df[["id", "dev_summary"]].to_dict(orient="records")
    }

    # 🔹 Prompt (your improved version)
    prompt = f"""
    You are a real estate expert.

    You are given:

    1. RAW PROPERTY DATA including:
        - price value insight (analysis_msg)
        - risk profile (risk_label)
        - growth potential (growth_label, growth_reason)
        - rental performance (monthly rent, rental yield, investment rating)
    2. SCORING DATA (price_score, risk_score, growth_score)
    3. DEVELOPMENT DATA (latest infrastructure insights if available)

    {context}

    Instructions:
    - Identify the BEST property
    - Use REAL insights from RAW data:
    - location quality
    - infrastructure / future growth
    - price value (not just number, but value)
    - risk factors
    - Use fields like analysis_msg, growth_reason if available
    - DO NOT mention or rely on scores directly
    - DO NOT repeat raw numbers blindly
    - Be practical and human-like
    - Use development_data if available for growth comparison
    - Prefer real-time developments over static growth_reason
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