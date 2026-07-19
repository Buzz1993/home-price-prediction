# ===============================
# advisor_node.py
# ===============================

from src.agents.advisor_agent import run_advisor_agent
from src.llm.provider_router import ask_llm


# =========================================
# FIND PROPERTY USING PROPERTY ID
# =========================================
def find_property_by_id(
    advisor_df,
    user_msg
):
    """
    Find property row if user mentioned
    property ID in message.
    """

    for _, row in advisor_df.iterrows():

        row_id = str(
            row.get("id", "")
        ).strip().lower()

        if row_id and row_id in user_msg:
            return row

    return None


# =========================================
# FIND BEST PROPERTY
# =========================================
def find_best_property(
    advisor_df
):
    """
    Return property with highest
    overall_score.
    """

    if advisor_df.empty:
        return None

    return advisor_df.sort_values(
        "overall_score",
        ascending=False
    ).iloc[0]


# =========================================
# MAIN NODE
# =========================================
def advisor_node(state):
    """
    Final investment advisor.

    Uses comparison_result as the
    single source of truth.
    """

    print("✅ advisor_node executed")

    comparison_raw = state.get(
        "comparison_raw"
    )

    comparison_result = state.get(
        "comparison_result"
    )

    # =====================================
    # VALIDATION
    # =====================================
    #prevent advisor_node from crashing when comparison data is missing.
    if (
        comparison_raw is None
        or comparison_raw.empty
    ):

        state["response"] = (
            "Please compare properties first."
        )

        return state

    if (
        comparison_result is None
        or comparison_result.empty
    ):

        state["response"] = (
            "Comparison results not available."
        )

        return state

    # =====================================
    # BUILD ADVISOR DATA
    # =====================================

    advisor_df = run_advisor_agent(
        comparison_raw
    )

    advisor_df = advisor_df.merge(
        comparison_result[
            [
                "id",
                "overall_score",
                "verdict",
                "comparison_reason"
            ]
        ],
        on="id",
        how="left"
    )

    # =====================================
    # DETECT TARGET PROPERTY
    # =====================================

    user_msg = state.get(
        "user_message",
        ""
    ).lower()

    property_row = find_property_by_id(
        advisor_df,
        user_msg
    )

    # -------------------------------------
    # BEST PROPERTY REQUEST
    # -------------------------------------

    if property_row is None:

        if (
            "winner" in user_msg
            or "best property" in user_msg
            or "should i buy" in user_msg
            or "which property should i buy" in user_msg
        ):

            property_row = find_best_property(advisor_df) # get property with highest overall score

    # -------------------------------------
    # DEFAULT
    # -------------------------------------

    if property_row is None:

        property_row = find_best_property(advisor_df)

    # -------------------------------------
    # KEEP ONLY ONE PROPERTY
    # -------------------------------------

    advisor_df = advisor_df[advisor_df["id"] == property_row["id"]] # filter to single property

    # =====================================
    # BUILD PROPERTY CONTEXT
    # =====================================
    property_text = ""
    for _, row in advisor_df.iterrows():
        property_text += f"""
PROPERTY ID:
{row.get('id')}

OVERALL SCORE:
{row.get('overall_score')}

VERDICT:
{row.get('verdict')}

COMPARISON REASON:
{row.get('comparison_reason')}

VALUATION:
{row.get('analysis_flag')} ({row.get('price_position')})

VALUATION DETAIL:
{row.get('analysis_msg')}

RISK PROFILE:
{row.get('risk_label')} (Score: {row.get('risk_score')})

GROWTH OUTLOOK:
{row.get('growth_label')}

GROWTH DETAILS:
{row.get('growth_reason')}

INFRASTRUCTURE / DEVELOPMENT SUMMARY:
{row.get('dev_summary')}

RENTAL PROFILE:
Rating: {row.get('investment_rating')} | Yield: {row.get('rental_yield_percent')} | Demand: {row.get('demand_level')}

RENTAL STRATEGY:
{row.get('rental_strategy')}

NEGOTIATION POWER:
Power: {row.get('negotiation_power')} | Score: {row.get('negotiation_score')} | Suggested Discount: {row.get('suggested_discount_percent')}

TARGET ENTRY PRICE:
{row.get('target_price')}
--------------------------------
"""

    # Gated Cross-Property Explanation Strategy
    comparison_explanation = state.get("explanation")
    if not comparison_explanation:
        comparison_explanation = "No explanation available."

    advisor_context = property_text

    if any(
        k in user_msg
        for k in [
            "should i buy",
            "winner",
            "best property",
            "which property should i buy",
        ]
    ):
        advisor_context += f"\n\nCOMPARISON EXPLANATION:\n{comparison_explanation}"

    # =====================================
    # PROMPT
    # =====================================
    print("\n=== PROPERTY TEXT ===")
    print(property_text)

    prompt = f"""
You are an expert Indian real estate advisor.

The user asked:

{user_msg}

Analyze ONLY the provided data.

DATA:

{advisor_context}

IMPORTANT:

Answer ONLY for the property shown above.

Use ONLY values provided.

Do NOT invent:

* appreciation %
* future returns
* CAGR
* growth multipliers
* valuation estimates
* infrastructure projects
* timelines
* market forecasts
* future price predictions
* additional scores
* additional calculations
* additional risks
* additional positives
* buyer profiles not explicitly supported by data

If information is missing, say:

"Data not available."

VERY IMPORTANT:

You are NOT allowed to create a new recommendation.

Use the existing verdict only.

Decision Mapping:

🏆 Best Value -> BUY
💎 Undervalued -> BUY
💰 Strong Investment -> CONSIDER
👍 Balanced -> CONSIDER
🚀 High Growth -> CONSIDER
⚠️ Risky -> AVOID
💸 Expensive -> AVOID

Never override the verdict.

TASK:

1. Explain why the verdict was assigned
2. Show exact overall score
3. List favorable signals directly present in the data
4. List risk signals directly present in the data
5. Explain rental profile using only provided values
6. Explain growth profile using only provided values

REASONING PRIORITY:

1. comparison_reason
2. verdict
3. valuation
4. risk profile
5. growth profile

Use comparison_reason as the primary explanation for why the verdict was assigned.

RULES FOR POSITIVES:

Only use information explicitly present in:

* comparison_reason
* growth_label
* growth_reason
* investment_rating
* rental_strategy
* negotiation_power
* risk_label
* dev_summary

Valuation handling:

* undervalued = positive
* fair = neutral
* overpriced = NOT a positive

High negotiation power may be listed as a positive.

Do not create new positives.

RULES FOR RISKS:

Risk signals may only come from:

* risk_label
* risk_score
* analysis_flag
* rental profile
* negotiation profile

Valuation handling:

* overpriced = risk
* fair = neutral
* undervalued = not a risk

Do not infer risks.

Do NOT generate risks from missing information.

Missing information is NOT a risk.

Do not mention:

* market saturation
* future uncertainty
* execution risk
* macro risks
* locality concerns

unless explicitly provided in the data.

If no explicit risks are present, write:

* No major risk signals detected.

If a section has no valid items, write:

* None identified from provided data.

If the user asks for exact scores, show exact values from data only.

OUTPUT FORMAT:

🏠 Property: <id>

Decision:

* BUY / CONSIDER / AVOID

Verdict:

* exact verdict

Overall Score:

* exact score

Positives:

* bullet points using only provided data

Risks:

* bullet points using only provided data

Reason:

* concise explanation using only provided data
  """

    state["response"] = ask_llm(
        prompt
    )

    return state