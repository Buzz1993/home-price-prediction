#src/services/chat_service.py
# ===============================
# chat_service.py
# ===============================

from src.llm.deepseek_client import ask_deepseek_stream
from src.llm.memory_store import SQLiteMemoryStore
from src.llm.deepseek_memory import extract_memory

memory_store = SQLiteMemoryStore()
USER_ID = "default_user"


# =============================
# BUILD CONTEXT
# =============================
def build_context(
    recs,
    comparison_result,
    comparison_raw=None,
    last_explanation=None
):
    
    """
    Build property context for the LLM.
    """

    sections = []

    # =====================================
    # INPUT PROPERTY
    # =====================================
    if recs is not None and "input" in recs:

        input_df = recs["input"].copy()

        if not input_df.empty:

            sections.append(
                "INPUT PROPERTY:\n" +
                input_df.to_string(index=False)
            )

    # =====================================
    # SIMILAR PROPERTIES
    # =====================================
    if recs is not None and "similar" in recs:

        sim_df = recs["similar"].copy()

        if not sim_df.empty:

            sections.append(
                "SIMILAR PROPERTIES:\n" +
                sim_df.to_string(index=False)
            )


    # =====================================
    # DETAILED PROPERTY DATA
    # =====================================
    if comparison_raw is not None and not comparison_raw.empty:

        important_cols = [
            "id",
            "project_name",
            "location",
            "price",
            "area",
            "bhk_type",
            "locality_rating",
            "risk_score",
            "growth_score",
            "hybrid_score",
            "rental_yield_percent",
            "investment_rating",
            "negotiation_power",
            "target_price",
            "price_position",
            "why_recommended"
        ]

        available_cols = [
            c for c in important_cols
            if c in comparison_raw.columns
        ]

        sections.append(
            "DETAILED PROPERTY DATA:\n" +
            comparison_raw[available_cols].to_string(index=False)
        )

    # =====================================
    # COMPARISON RESULT
    # =====================================
    if comparison_result is not None and not comparison_result.empty:

        compare_df = comparison_result.copy()

        if "price" in compare_df.columns:
            compare_df["price"] = compare_df["price"].apply(
                lambda x: f"₹{x} Cr"
            )

        sections.append(
            "COMPARISON RESULT:\n" +
            compare_df.to_string(index=False)
        )

    # =====================================
    # RENTAL ESTIMATE
    # =====================================
    if comparison_raw is not None and not comparison_raw.empty:

        rent_cols = [
            c for c in [
                "id",
                "area",
                "min_rent",
                "max_rent",
                "monthly_rent_estimate",
                "rental_yield_percent"
            ]
            if c in comparison_raw.columns
        ]

        if rent_cols:

            sections.append(
                "RENTAL ESTIMATE:\n" +
                comparison_raw[rent_cols].to_string(index=False)
            )

    # =====================================
    # PROPERTY INSIGHTS
    # =====================================
    if last_explanation:

        sections.append(
            "PROPERTY INSIGHTS:\n" +
            str(last_explanation)
        )

    # =====================================
    # FINAL
    # =====================================
    if not sections:
        return "No property data available"

    return "\n\n".join(sections)


# =============================
# DETECT INTENT
# =============================
def detect_intent(user_msg):
    # =============================
    # Identify user intent
    # =============================
    """
    Detects if query is rent-related or comparison-related.
    """

    msg = user_msg.lower()

    is_rent = any(word in msg for word in ["rent", "rental", "income"])
    is_compare = any(word in msg for word in ["compare", "better", "best"])

    return is_rent, is_compare


# =============================
# HANDLE RENT RESPONSE
# =============================
def generate_rent_response(recs):
    # =============================
    # Build rental analysis text
    # =============================
    """
    Generates rental insights from similar properties.
    """

    if recs is None:
        return "No rental data available."

    rent_df = recs["similar"].copy().head(5)

    response = "🏠 Rental Analysis:\n\n"

    for _, row in rent_df.iterrows():
        yield_ = row.get("rental_yield_percent") or "0%"
        demand = row.get("demand_level") or "Unknown"
        rating = row.get("investment_rating") or "Unknown"
        
        # =============================
        # Safe rent handling
        # =============================
        rent = row.get("monthly_rent_estimate") or 0

        try:
            rent = int(float(rent))
        except:
            rent = 0

        response += (
            f"📍 {row['id']} ({row['location']})\n"
            f"• Rent: ₹{rent:,}/month\n"
            f"• Yield: {yield_}\n"
            f"• Demand: {demand}\n"
            f"• Rating: {rating}\n\n"
        )

    return response


# =============================
# STREAM LLM RESPONSE
# =============================
def stream_llm_response(
        user_msg,
        recs,
        comparison_result,
        comparison_raw=None,
        last_explanation=None,
        history=None
    ):
    # =============================
    # Generate streaming response
    # =============================
    """
    Handles full chat pipeline including memory and LLM streaming.
    """

    # MEMORY
    mem = extract_memory(user_msg)
    if mem:
        memory_store.add_memory(USER_ID, mem)

    context = build_context(
        recs,
        comparison_result,
        comparison_raw,
        last_explanation
    )

    history_text = ""

    if history:
        for role, msg in history[-10:]:
            history_text += f"{role}: {msg}\n"


    prompt = f"""
    You are an expert real estate assistant with strong memory.


    You MUST remember and use:
    - Input property
    - Similar properties
    - Detailed property comparison data
    - Comparison results
    - Rental estimates
    - Property insights
    - Previous user questions
    - Previous assistant answers

    IMPORTANT RULES:
    - NEVER hallucinate
    - Use ONLY provided property data
    - If user asks for IDs of similar properties,
    answer from SIMILAR PROPERTIES section
    - If user asks for compared properties, answer 
    from DETAILED PROPERTY DATA or COMPARISON RESULT
    - Maintain conversation continuity
    - Answer directly
    - IMPORTANT: Prices are in Indian Rupees (₹)
    - NEVER use dollars ($)
    - Format like ₹1.7 Cr

    CHAT HISTORY:
    {history_text}

    PROPERTY DATA:
    {context}

    USER QUESTION:
    {user_msg}

    ASSISTANT:
    """

    return ask_deepseek_stream(prompt)