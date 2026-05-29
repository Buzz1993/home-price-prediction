# ===============================
# system_chat_node.py
# ===============================

from src.llm.deepseek_client import ask_deepseek


APPLICATION_OVERVIEW = """
PROJECT OVERVIEW

This is an AI-powered real estate analysis platform.

MAIN MODULES

1. Property Recommendation
   - Content-based filtering
   - Hybrid weighted ranking
   - Recommendation explanations

2. Property Comparison
   - Multi-property comparison
   - Overall scoring
   - Verdict generation

3. Rental Analysis
   - Monthly rent estimation
   - Annual rent estimation
   - Rental yield calculation
   - Investment rating

4. Risk Analysis
   - Risk scoring
   - Risk categorization
   - Risk labels

5. Growth Analysis
   - Future growth scoring
   - Infrastructure signal detection
   - Growth insights

6. Negotiation Analysis
   - Negotiation power
   - Suggested discount
   - Target buying price

7. Property Valuation
   - Overpriced detection
   - Undervalued detection
   - Fair value analysis

8. Home Price Prediction
   - ML model prediction
   - FastAPI prediction service
   - MLflow model registry
   - DagsHub integration

9. AI Assistant
   - Property Q&A
   - Comparison explanations
   - Project/workflow explanations

WORKFLOW

Search
→ Recommendation
→ Property Selection
→ Comparison
→ Explanation
→ AI Chat

AVAILABLE AGENTS

- comparison_agent
- analysis_agent
- rental_agent
- risk_agent
- future_agent
- negotiation_agent
- explanation_agent

IMPORTANT

Only discuss functionality listed above.

Do not invent:
- integrations
- APIs
- CRM systems
- subscriptions
- dashboards
- blockchain features
- REIT features
- future functionality
"""


def system_chat_node(state):
    """
    Handles project, workflow,
    architecture and implementation questions.
    """

    print("✅ system_chat_node executed")

    user_msg = state.get(
        "user_message",
        ""
    )

    prompt = f"""
You are a senior software architect.

{APPLICATION_OVERVIEW}

USER QUESTION:
{user_msg}

RULES

- Answer ONLY using PROJECT OVERVIEW.
- Never invent functionality.
- Never invent modules.
- Never invent APIs.
- Never invent databases.
- Never invent integrations.
- Never invent dashboards.
- Never invent future features.

If the user asks about something that is not listed
in PROJECT OVERVIEW, reply:

"That functionality is not currently implemented."

Keep answers concise and accurate.

ANSWER:
"""

    response = ask_deepseek(prompt)

    state["response"] = response

    return state