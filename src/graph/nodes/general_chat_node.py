# ===============================
# general_chat_node.py
# ===============================

from src.llm.deepseek_client import ask_deepseek

from src.services.chat_service import build_context


def general_chat_node(state):
    print("✅ general_chat_node executed")

    context = build_context(
        state.get("recommendations"),
        state.get("selected_properties"),
        state.get("comparison_result"),
        state.get("comparison_raw"),
        state.get("explanation")
    )

    prompt = f"""
    You are an expert real estate assistant.

    MEMORY:
    {state.get("memory")}

    PROPERTY CONTEXT:
    {context}

    USER:
    {state["user_message"]}

    IMPORTANT:
    - Use only provided property data
    - Do not hallucinate
    - Prices are in INR
    """

    response = ask_deepseek(prompt)

    state["response"] = response

    return state