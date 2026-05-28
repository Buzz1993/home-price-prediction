# ===============================
# rental_node.py
# ===============================

from src.agents.rental_agent import run_rental_agent
from src.llm.deepseek_client import ask_deepseek


def rental_node(state):
    """
    Analyzes property rental data
    and generates AI-based rental
    investment insights and advice.
    """

    print("✅ rental_node executed")

    selected_df = state.get("selected_properties")

    # ---------------------------------
    # VALIDATION
    # ---------------------------------
    if selected_df is None or selected_df.empty:

        state["response"] = (
            "Please select at least one property first."
        )

        return state

    # ---------------------------------
    # RUN RENTAL ANALYSIS
    # ---------------------------------
    rental_df = run_rental_agent(selected_df)

    # ---------------------------------
    # BUILD CONTEXT
    # ---------------------------------
    property_text = ""

    for _, row in rental_df.iterrows():

        property_text += f"""

        PROPERTY ID: {row.get('id')}

        Monthly Rent Estimate:
        ₹{row.get('monthly_rent_estimate')}

        Annual Rent:
        ₹{row.get('annual_rent')}

        Rental Yield:
        {row.get('rental_yield_percent')}

        Demand Level:
        {row.get('demand_level')}

        Investment Rating:
        {row.get('investment_rating')}

        Rental Strategy:
        {row.get('rental_strategy')}

        --------------------------------
        """

    # ---------------------------------
    # LLM PROMPT
    # ---------------------------------
    prompt = f"""
    You are an expert Indian real estate rental advisor.

    Analyze the following rental data carefully.

    {property_text}

    TASK:
    Give practical rental investment advice.

    IMPORTANT RULES:
    - Use simple human language
    - Sound like an experienced rental advisor
    - Explain rental income potential
    - Mention rental yield
    - Mention tenant demand
    - Mention investment quality
    - Mention rental risks if applicable
    - Explain whether property is good for:
        - rental investment
        - appreciation
        - self use
    - Use bullet points
    - Keep response practical and concise
    - Prices are in INR
    - NEVER use dollars ($)
    - Format prices like ₹75,000/month

    OUTPUT FORMAT:

    🏠 Property: <id>

    Monthly Rent:
    - ₹xx/month

    Rental Yield:
    - xx%

    Demand:
    - High / Medium / Low

    Investment Quality:
    - Excellent / Good / Average / Low

    Rental Strategy:
    - short practical explanation

    Final Advice:
    - short practical conclusion

    -----------------------------
    """

    # ---------------------------------
    # GENERATE RESPONSE
    # ---------------------------------
    response = ask_deepseek(prompt)

    state["response"] = response

    return state