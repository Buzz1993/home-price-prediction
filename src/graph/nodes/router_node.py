# # ===============================
# # router_node.py
# # ===============================

# # router_node sets state["route"] based on user query. Then add_conditional_edges(from workflow.py) uses that route value to execute the matching workflow/node.

# def router_node(state):
#     """
#     Routes the user query to the correct
#     chat workflow like rental, prediction,
#     negotiation, valuation, or general chat.
#     """

#     print("✅ router_node executed")

#     msg = state["user_message"].lower()

#     # -----------------------------
#     # RENTAL
#     # -----------------------------
#     if any(word in msg for word in [
#         "rent",
#         "rental",
#         "tenant",
#         "lease",
#         "yield",
#         "income",
#         "monthly rent",
#         "rental estimate"
#     ]):

#         state["route"] = "rental"


#     # -----------------------------
#     # PRICE PREDICTION
#     # -----------------------------
#     elif any(word in msg for word in [
#         "predict",
#         "prediction",
#         "estimated price",
#         "estimate",
#         "price prediction",
#         "what should this cost",
#         "property value"
#     ]):

#         state["route"] = "prediction"

#     # -----------------------------
#     # NEGOTIATION
#     # -----------------------------
#     elif any(word in msg for word in [
#         "negotiate",
#         "negotiable",
#         "negotiation",
#         "discount",
#         "best price",
#         "reduce price",
#         "deal"
#     ]):

#         state["route"] = "negotiation"

#     # -----------------------------
#     # VALUATION
#     # -----------------------------
#     elif any(word in msg for word in [
#         "overpriced",
#         "undervalued",
#         "valuation",
#         "worth",
#         "fair price"
#     ]):

#         state["route"] = "valuation"

#     # -----------------------------
#     # GENERAL CHAT
#     # -----------------------------
#     else:

#         state["route"] = "general"

#     # -----------------------------
#     # DEBUG ROUTE
#     # -----------------------------
#     print(f"➡️ Routed to: {state['route']}")

#     return state

#===================================================================================================================================================================================

# ===============================
# router_node.py
# ===============================

# router_node sets state["route"] based on user query.
# workflow.py then uses add_conditional_edges()
# to execute the correct node.

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
    "implemented",
    "current modules",
    "current agents",
    "overview",
    "project overview",
    "how does this work",
    "what does this project do",
    "what is this project",
    "what are the agents",
    "how is rental calculated",
    "how is risk calculated",
    "how is growth calculated",
    "how is comparison done",
    "how does recommendation work",
    "how is price predicted",
    "valuation logic"
]


def router_node(state):
    """
    Route user query to the correct workflow.

    Available routes:
    - system
    - rental
    - prediction
    - negotiation
    - valuation
    - general
    """

    print("✅ router_node executed")

    msg = state.get(
        "user_message",
        ""
    ).lower()

    # -----------------------------
    # SYSTEM / PROJECT QUESTIONS
    # -----------------------------
    if any(
        keyword in msg
        for keyword in SYSTEM_KEYWORDS
    ):

        state["route"] = "system"

    # -----------------------------
    # RENTAL
    # -----------------------------
    elif any(word in msg for word in [
        "rent",
        "rental",
        "tenant",
        "lease",
        "yield",
        "rental yield",
        "monthly rent",
        "annual rent",
        "rental estimate",
        "rental income",
        "income property"
    ]):

        state["route"] = "rental"

    # -----------------------------
    # PRICE PREDICTION
    # -----------------------------
    elif any(word in msg for word in [
        "predict",
        "prediction",
        "predicted price",
        "estimated price",
        "price estimate",
        "price prediction",
        "property value",
        "future price",
        "what should this cost"
    ]):

        state["route"] = "prediction"

    # -----------------------------
    # NEGOTIATION
    # -----------------------------
    elif any(word in msg for word in [
        "negotiate",
        "negotiation",
        "negotiable",
        "discount",
        "best price",
        "reduce price",
        "target price",
        "deal",
        "bargain"
    ]):

        state["route"] = "negotiation"

    # -----------------------------
    # VALUATION
    # -----------------------------
    elif any(word in msg for word in [
        "overpriced",
        "undervalued",
        "valuation",
        "fair value",
        "fair price",
        "worth buying",
        "worth it",
        "market value"
    ]):

        state["route"] = "valuation"

    # -----------------------------
    # GENERAL PROPERTY CHAT
    # -----------------------------
    else:

        state["route"] = "general"

    # -----------------------------
    # DEBUG
    # -----------------------------
    print(f"➡️ Routed to: {state['route']}")

    return state