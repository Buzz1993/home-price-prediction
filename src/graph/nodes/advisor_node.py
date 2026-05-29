# ===============================
# advisor_node.py
# ===============================

from src.agents.advisor_agent import run_advisor_agent
from src.llm.deepseek_client import ask_deepseek


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

            property_row = find_best_property(
                advisor_df
            )

    # -------------------------------------
    # DEFAULT
    # -------------------------------------

    if property_row is None:

        property_row = find_best_property(
            advisor_df
        )

    # -------------------------------------
    # KEEP ONLY ONE PROPERTY
    # -------------------------------------

    advisor_df = advisor_df[
        advisor_df["id"]
        ==
        property_row["id"]
    ]

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

SUITABLE FOR:
{row.get('suitable_for')}

POSITIVES:
{row.get('positives')}

RISKS:
{row.get('risks')}

--------------------------------
"""

    # =====================================
    # PROMPT
    # =====================================

    prompt = f"""
You are an expert Indian real estate advisor.

The user asked:

{user_msg}

Analyze ONLY the provided data.

DATA:

{property_text}

IMPORTANT:

Answer ONLY for the property shown above.

Use ONLY values provided.

Do NOT invent:

- appreciation %
- future returns
- CAGR
- growth multipliers
- valuation estimates
- infrastructure projects
- timelines
- market forecasts
- future price predictions
- additional scores
- additional calculations

If information is missing say:

"Data not available."

VERY IMPORTANT:

You are NOT allowed to create
a new recommendation.

Use the existing verdict only.

Mapping:

🏆 Best Value -> BUY

💰 Strong Investment -> CONSIDER

👍 Balanced -> CONSIDER

⚠️ Risky -> AVOID

💸 Expensive -> AVOID

Never override the verdict.

TASK:

1. Explain why the verdict was assigned
2. Show exact overall score
3. Show positives
4. Show risks
5. Explain who the property is suitable for

If the user asks for exact scores,
show exact values from data only.

OUTPUT FORMAT:

🏠 Property: <id>

Verdict:
- exact verdict

Overall Score:
- exact score

Suitable For:
- exact value

Positives:
- bullet points

Risks:
- bullet points

Reason:
- explain using only provided data
"""

    state["response"] = ask_deepseek(
        prompt
    )

    return state