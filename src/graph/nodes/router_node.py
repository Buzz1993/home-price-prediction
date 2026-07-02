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

ADVISOR_KEYWORDS = [

    "should i buy",

    "buy this property",

    "is this a good investment",

    "investment advice",

    "recommendation",

    "final advice",

    "which property should i buy",

    "best investment",

    "best property",

    "advisor",

    "why did you mark",

    "why buy",

    "show exact scores",

    "show scores",

    "advisor score",

    "buy decision",

    "investment decision",

    "recommendation reason",

    "why recommendation"
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
    # ADVISOR
    # -----------------------------
    elif any(
        keyword in msg
        for keyword in ADVISOR_KEYWORDS
    ):

        state["route"] = "advisor"

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