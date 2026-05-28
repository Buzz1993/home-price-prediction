# ===============================
# negotiation_node.py
# ===============================

from src.agents.negotiation_agent import run_negotiation_agent
from src.llm.deepseek_client import ask_deepseek


def negotiation_node(state):
    """
    Analyzes property negotiation data
    and generates AI-based negotiation
    strategies and buyer advice.
    """

    print("✅ negotiation_node executed")

    comparison_raw = state.get("comparison_raw")
    comparison_result = state.get("comparison_result")
    # print("===============================")
    # print("comarison_raw columns", state.get("comparison_raw").columns.tolist())
    # print("comparison_result columns", state.get("comparison_result").columns.tolist())
    # print("===============================")


    # ---------------------------------
    # VALIDATION
    # ---------------------------------
    if comparison_raw is None or comparison_raw.empty:

        state["response"] = (
            "Please select at least one property first."
        )

        return state
    
    # ---------------------------------
    # MERGE COMPARISON INSIGHTS
    # ---------------------------------
    important_cols = [
        "id",
        "overall_score",
        "verdict",
        "comparison_reason"
    ]

    # ---------------------------------
    # MERGE COMPARISON INSIGHTS
    # ---------------------------------
    if (
        comparison_result is not None
        and not comparison_result.empty
    ):

        important_cols = [
            "id",
            "overall_score",
            "verdict",
            "comparison_reason"
        ]

        comparison_merge = comparison_result[important_cols]

        negotiation_input_df = comparison_raw.merge(
            comparison_merge,
            on="id",
            how="left"
        )

    else:
        negotiation_input_df = comparison_raw.copy()

    # ---------------------------------
    # RUN NEGOTIATION ANALYSIS
    # ---------------------------------
    negotiation_df = run_negotiation_agent(
        negotiation_input_df
    )

    # ---------------------------------
    # BUILD CONTEXT
    # ---------------------------------
    property_text = ""

    for _, row in negotiation_df.iterrows():

        property_text += f"""

        PROPERTY ID: {row.get('id')}

        Negotiation Power: {row.get('negotiation_power')}

        Suggested Discount: {row.get('suggested_discount_percent')}

        Target Price: ₹{row.get('target_price')} Cr

        Price Position: {row.get('price_position')}

        Strategy:
        {row.get('strategy')}

        Talking Points:
        {row.get('talking_points')}

        --------------------------------
        """

    # ---------------------------------
    # LLM PROMPT
    # ---------------------------------
    prompt = f"""
    You are an expert Indian real estate negotiation advisor.

    Analyze the following negotiation data carefully.

    {property_text}

    TASK:
    Give practical negotiation advice for the buyer.

    IMPORTANT RULES:
    - Use simple human language
    - Sound like an experienced property negotiator
    - Explain whether strong negotiation is possible
    - Mention expected discount range
    - Mention negotiation leverage
    - Mention risks if applicable
    - Mention if price already looks fair
    - Use bullet points
    - Keep response practical and concise
    - Prices are in INR
    - NEVER use dollars ($)
    - Format prices like ₹2.45 Cr

    OUTPUT FORMAT:

    🏠 Property: <id>

    Negotiation Strength:
    - Strong / Medium / Low

    Expected Discount:
    - xx%

    Best Strategy:
    - bullet points

    Talking Points:
    - bullet points

    Final Advice:
    - short practical conclusion
    """

    # ---------------------------------
    # GENERATE RESPONSE
    # ---------------------------------
    response = ask_deepseek(prompt)

    state["response"] = response

    return state