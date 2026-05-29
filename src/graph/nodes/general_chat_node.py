# ===============================
# general_chat_node.py
# ===============================

from src.llm.deepseek_client import ask_deepseek
from src.services.chat_service import build_context


SYSTEM_KEYWORDS = [
    "application",
    "app",
    "software",
    "system",
    "workflow",
    "architecture",
    "agent",
    "agents",
    "feature",
    "features",
    "project",
    "platform",
    "module",
    "modules",
    "component",
    "components",
    "how does this work",
    "what does this do",
    "what is this project",
    "what are the agents",
    "how is rental calculated",
    "how is risk calculated",
    "how is growth calculated",
    "how is comparison done",
    "how does recommendation work",
    "how is price predicted",
    "price prediction",
    "valuation logic"
]


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
Do not invent integrations,
APIs,
CRM systems,
subscriptions,
REIT features,
blockchain features,
dashboards,
or functionality not implemented.
"""


def general_chat_node(state):
    """
    Handles:
    - Property questions
    - Recommendation questions
    - Comparison questions
    - Architecture questions
    - Workflow questions
    """

    print("✅ general_chat_node executed")

    user_msg = state.get(
        "user_message",
        ""
    )

    context = build_context(
        state.get("recommendations"),
        state.get("comparison_result"),
        state.get("comparison_raw"),
        state.get("explanation")
    )

    is_system_question = any(
        keyword in user_msg.lower()
        for keyword in SYSTEM_KEYWORDS
    )

    if is_system_question:

        prompt = f"""
You are a senior software architect.

{APPLICATION_OVERVIEW}

USER QUESTION:
{user_msg}

RULES

- Answer ONLY from PROJECT OVERVIEW.
- Never invent functionality.
- Never invent databases.
- Never invent APIs.
- Never invent dashboards.
- Never invent integrations.
- Never invent future features.
- If functionality is unavailable say:

"That functionality is not currently implemented."

Give concise and accurate answers.

ANSWER:
"""

    else:

        prompt = f"""
You are an expert real-estate advisor.

{APPLICATION_OVERVIEW}

RULES

- Use ONLY provided property data.
- Never hallucinate.
- Never invent properties.
- Never invent IDs.
- Never invent prices.
- Use all matching properties.
- Prices are in INR.
- Format prices as ₹2.35 Cr.
- If information is missing, clearly say so.
- Use recommendation, rental, risk,
  growth, valuation and negotiation
  information whenever available.
- If user asks for comparison,
  compare all available properties.
- If user asks for table,
  return markdown table.

USER MEMORY:
{state.get("memory", [])}

PROPERTY CONTEXT:
{context}

USER QUESTION:
{user_msg}

ANSWER:
"""

    response = ask_deepseek(prompt)

    state["response"] = response

    return state