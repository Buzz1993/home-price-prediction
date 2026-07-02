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

    comparison_raw = state.get("comparison_raw")
    comparison_result = state.get("comparison_result")

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

        comparison_merge = comparison_result[
            important_cols
        ]

        rental_input_df = comparison_raw.merge(
            comparison_merge,
            on="id",
            how="left"
        )

    else:
        rental_input_df = comparison_raw.copy()

    # ---------------------------------
    # RUN RENTAL ANALYSIS
    # ---------------------------------
    rental_df = run_rental_agent(rental_input_df)

    # ---------------------------------
    # BUILD CONTEXT
    # ---------------------------------
    property_text = ""

    for _, row in rental_df.iterrows():

        property_text += f"""

        PROPERTY ID: {row.get('id')}

        Current Price:
        ₹{row.get('current_price')} Cr

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

    Analyze the following rental data.

    {property_text}

    TASK:

    Summarize the rental profile of each property using ONLY the provided PROPERTY DATA.

    IMPORTANT RULES:

    * Use simple human language.
    * Use concise bullet points.
    * Keep the response practical and short.
    * Use ONLY information provided in PROPERTY DATA.
    * Every statement must be directly supported by PROPERTY DATA.
    * Do NOT introduce new facts, assumptions, risks, calculations, recommendations, or opinions.
    * Do NOT speculate.
    * Do NOT infer missing information.

    PRICE RULES:

    * Prices are in INR.
    * Never use dollars ($).
    * Format prices like ₹3.35 Cr.
    * Format rent like ₹75,000/month.

    RENTAL RULES:

    Mention only:

    * Current Price
    * Monthly Rent
    * Annual Rent
    * Rental Yield
    * Demand Level
    * Investment Rating
    * Rental Strategy
    * Use Rental Strategy exactly as provided or lightly rephrase it for readability.
    * Do NOT add additional strategy points.
    * Do NOT compare properties.
    * Do NOT rank properties.

    FORBIDDEN UNLESS EXPLICITLY PROVIDED:

    * vacancy rates
    * tenant turnover
    * occupancy levels
    * rental market averages
    * location quality
    * appreciation potential
    * resale potential
    * long-term holding suitability
    * short-term flipping suitability
    * investor suitability
    * self-use suitability
    * future price appreciation
    * cash flow projections
    * market forecasts
    * additional risks
    * profitability claims
    * investment recommendations

    If a topic is not explicitly present in PROPERTY DATA, omit it entirely.

    SUMMARY RULES:

    * Generate Summary using ONLY:
        * Demand Level
        * Investment Rating
        * Rental Strategy
    * Do NOT introduce any new opinion.
    * Do NOT introduce any recommendation.
    * Do NOT introduce any risk assessment.
    * Do NOT introduce any investment judgment.
    * Simply restate the provided data in one sentence.

    OUTPUT FORMAT:

    🏠 Property: <id>

    Current Price:

    * ₹x.xx Cr

    Monthly Rent:

    * ₹xx/month

    Annual Rent:

    * ₹xx/year

    Rental Yield:

    * xx%

    Demand:

    * High / Medium / Low

    Investment Quality:

    * Excellent / Good / Average / Low

    Rental Strategy:

    * <provided rental strategy>

    Summary:

    * One factual sentence based only on Demand Level, Investment Rating, and Rental Strategy.

    ---

    """


    # ---------------------------------
    # GENERATE RESPONSE
    # ---------------------------------
    response = ask_deepseek(prompt)

    state["response"] = response

    return state