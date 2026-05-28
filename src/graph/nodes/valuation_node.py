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
    """
    Analyzes property valuation using
    comparison data and generates
    AI-based pricing insights.
    """

    print("✅ valuation_node executed")
    # print("===============================")
    # print("comarison_raw columns", state.get("comparison_raw").columns.tolist())
    # print("comparison_result columns", state.get("comparison_result").columns.tolist())
    # print("===============================")

    comparison_result = state.get("comparison_result")

    comparison_raw = state.get("comparison_raw")

    # =================================
    # MERGE SUMMARY + DETAILED DATA
    # =================================
    comparison_df = comparison_result.merge(
        comparison_raw[
            [
                "id",
                "analysis_msg",
                "analysis_flag",
                "analysis_severity",
                "risk_label",
                "growth_label",
                "growth_reason",
                "price_position",
                "negotiation_power",
                "suggested_discount_percent"
            ]
        ],
        on="id",
        how="left"
    )

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

        comparison_reason = row.get(
            "comparison_reason",
            "No comparison insight available"
        )

        property_text += f"""
        PROPERTY ID: {pid}

        Price: {price}

        Risk Score: {risk}

        Growth Score: {growth}

        Valuation Insight: {analysis_msg}

        Verdict: {verdict}

        Comparison Insight: {comparison_reason}

        Growth Label: {row.get("growth_label")}

        Growth Reason: {row.get("growth_reason")}

        Risk Label: {row.get("risk_label")}

        Price Position: {row.get("price_position")}

        Negotiation Power: {row.get("negotiation_power")}

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
    - Use Growth Reason, Risk Label, and Price Position only as supporting context
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
    - Analyze EACH property independently
    - Do NOT compare unrelated properties unless explicitly required
    - Avoid repetitive wording across properties
    - Keep reasoning concise but slightly varied

    - If Valuation Insight says "Within fair price range", FINAL VERDICT MUST remain "Fair" unless explicit benchmark deviation is provided
    - Verdict labels like "Expensive" or "Best Value" are secondary context only and MUST NOT override Valuation Insight
    - NEVER invent or speculate about hidden risks, legal issues, market saturation, regulatory concerns, or future problems unless explicitly provided in PROPERTY DATA
    - Do NOT compare one property's price with another property unless explicitly required

    - Clearly explain the reason behind the valuation verdict
    - Use valuation deviation and pricing position as primary explanation
    - Explain WHY the property is overpriced, undervalued, or fair using the provided valuation insight and supporting context
    - Keep valuation reasoning concise (2-3 lines maximum)
    - Focus only on the strongest valuation reason
    - Keep valuation reasoning in short bullet points
    - Use 2-4 concise points maximum
    - Avoid long paragraphs

    OUTPUT FORMAT:

    PROPERTY: <id>

    VALUATION REASON:
    - Point 1
    - Point 2
    - Point 3

    FINAL VERDICT:
    <Overpriced / Undervalued / Fair / Cannot determine>

    -----------------------------
    """

    response = ask_deepseek(prompt)

    state["response"] = response

    return state