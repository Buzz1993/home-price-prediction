# # ===============================
# # general_chat_node.py
# # ===============================

# from src.llm.deepseek_client import ask_deepseek

# from src.services.chat_service import build_context


# def general_chat_node(state):
#     """
#     Handles general property-related chat
#     using property context, memory,
#     and user queries to generate AI responses.
#     """
#     print("✅ general_chat_node executed")

#     context = build_context(
#         state.get("recommendations"),
#         state.get("selected_properties"),
#         state.get("comparison_result"),
#         state.get("comparison_raw"),
#         state.get("explanation")
#     )

#     prompt = f"""
#     You are an expert real estate assistant.

#     MEMORY:
#     {state.get("memory")}

#     PROPERTY CONTEXT:
#     {context}

#     USER:
#     {state["user_message"]}

#     IMPORTANT:
#     - Use only provided property data
#     - Do not hallucinate
#     - Prices are in INR
#     """

#     response = ask_deepseek(prompt)

#     state["response"] = response

#     return state

#=====================================================================================================================================================================================

# ===============================
# general_chat_node.py
# ===============================

from src.llm.deepseek_client import ask_deepseek

from src.services.chat_service import build_context


def general_chat_node(state):
    """
    Handles general property-related chat
    using property context, memory,
    and user queries to generate AI responses.
    """
    print("✅ general_chat_node executed")

    context = build_context(
        state.get("recommendations"),
        state.get("comparison_result"),
        state.get("comparison_raw"),
        state.get("explanation")
    )

    # print("===============================")
    # print(type(state.get("recommendations")))
    # print(state.get("recommendations"))
    # print("comparison_result columns", state.get("comparison_result").columns.tolist())
    # print("comarison_raw columns", state.get("comparison_raw").columns.tolist())
    # print("explanation", state.get("explanation"))
    # print("===============================")


    prompt = f"""
    You are an expert Indian real estate assistant.

    You MUST carefully analyze ALL provided property data,
    including:
    - Input Property
    - Similar Properties
    - Detailed Property Comparison Data
    - Comparison Results
    - Rental Estimates
    - Property Insights
    - User Memory
    - Current User Question

    IMPORTANT RULES:
    - NEVER hallucinate
    - NEVER invent property IDs
    - NEVER skip matching properties
    - Use ONLY the provided property data
    - Carefully check ALL rows before answering
    - Prices are in INR (₹) and crores (Cr)
    - NEVER use dollars ($)
    - If user asks for tables, return clean markdown tables
    - If user asks for filtering/sorting/grouping,
    - Do not print any property data unless explicitly asked
    - Do not print any property if it does not match the criteria which user has not asked for.
    carefully analyze ALL property rows before answering

    USER MEMORY:
    {state.get("memory")}

    PROPERTY DATA:
    {context}

    USER QUESTION:
    {state["user_message"]}

    ASSISTANT:
    """

    response = ask_deepseek(prompt)

    state["response"] = response

    return state