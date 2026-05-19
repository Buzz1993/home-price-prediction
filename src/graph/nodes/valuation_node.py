# ===============================
# valuation_node.py
# ===============================

from src.llm.deepseek_client import ask_deepseek


def format_price(price):
    """
    Convert numeric price into readable INR crore format.
    """

    try:
        return f"₹{float(price):.2f} Cr"
    except:
        return str(price)


def valuation_node(state):

    print("✅ valuation_node executed")

    comparison_df = state.get("comparison_result")

    comparison_raw = state.get("comparison_raw")

    # ---------------------------------
    # SAFETY CHECK
    # ---------------------------------
    if comparison_df is None or comparison_df.empty:

        state["response"] = (
            "No comparison data available for valuation analysis."
        )

        return state

    # ---------------------------------
    # BUILD STRUCTURED PROPERTY CONTEXT
    # ---------------------------------
    property_text = ""

    for _, row in comparison_df.iterrows():

        pid = row.get("id", "Unknown")

        price = format_price(row.get("price", 0))

        risk = row.get("risk_score", "N/A")

        growth = row.get("growth_score", "N/A")


        analysis_msg = row.get(
            "analysis_msg",
            "Insufficient market benchmark data available"
        )

        verdict = row.get("verdict", "")

        explanation = row.get("explanation", "")

        property_text += f"""
        PROPERTY ID: {pid}

        Price: {price}

        Risk Score: {risk}

        Growth Score: {growth}

        Valuation Insight: {analysis_msg}

        Verdict: {verdict}

        Comparison Insight: {explanation}

        -----------------------------
        """

    # ---------------------------------
    # LLM PROMPT
    # ---------------------------------
    prompt = f"""
    You are an expert Indian real estate valuation analyst.

    Analyze the following properties carefully.

    PROPERTY DATA:
    {property_text}

    TASK:
    For EACH property:
    - Explain whether it is:
        - Overpriced
        - Undervalued
        - Fairly priced

    IMPORTANT RULES:
    - Use ONLY provided data
    - Use Valuation Insight as the PRIMARY and REQUIRED basis for price judgment
    - If Valuation Insight is unavailable, DO NOT classify property as overpriced, undervalued, or fair
    - In such cases, respond that valuation cannot be determined due to insufficient benchmark data
    - Do NOT use Verdict alone to decide valuation
    - Do NOT use overall score alone to decide valuation
    - Risk score and growth score are supporting context only, NOT valuation proof
    - Comparison Insight is ranking-related context only, NOT direct valuation evidence
    - Terms like "better price" do NOT automatically mean undervalued
    - Valuation Insight already contains the actual pricing deviation logic
    - Do NOT recalculate percentage deviation yourself
    - Do NOT reinterpret valuation formulas
    - Do NOT hallucinate
    - Do NOT assume:
        - storage units
        - flood zones
        - legal disputes
        - government projects
        - hidden risks
    - Prices are already in Indian Rupees (₹ Crores)
    - NEVER convert prices
    - NEVER guess missing information
    - Keep analysis practical and short
    - If valuation insight is unavailable, clearly state that market benchmark data is insufficient
    - Do NOT force a valuation judgment when supporting data is missing

    OUTPUT FORMAT:

    PROPERTY: <id>

    VALUATION:
    <short reasoning>

    FINAL VERDICT:
    - Overpriced
    - Undervalued
    - Fair
    - Cannot determine (if benchmark data unavailable)

    -----------------------------
    """

    response = ask_deepseek(prompt)

    state["response"] = response

    return state