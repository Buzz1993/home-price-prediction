# ===============================
# general_chat_node.py
# ===============================

from src.llm.deepseek_client import ask_deepseek
from src.services.chat_service_legacy import build_context


def general_chat_node(state):
    """
    Handles property-related conversations.

    Uses:
    - recommendations
    - comparison data
    - explanation data
    - user memory

    Generates answers using only
    available property information.
    """

    print("✅ general_chat_node executed")

    user_msg = state.get(
        "user_message",
        ""
    )

    print("===============================")
    print("recommendations", state.get("recommendations"))
    print("comarison_raw columns", state.get("comparison_raw").columns.tolist())
    print("comparison_result columns", state.get("comparison_result").columns.tolist())
    print("===============================")

    context = build_context(
        state.get("recommendations"),
        state.get("comparison_result"),
        state.get("comparison_raw"),
        state.get("explanation")
    )

    if "No property data available" in context:

        state["response"] = (
            "No property data is currently available. "
            "Please search or compare properties first."
        )

        return state

    prompt = f"""
You are an expert real-estate advisor.

RULES

- Use ONLY provided property data.
- Never hallucinate.
- Never invent properties.
- Never invent project names.
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