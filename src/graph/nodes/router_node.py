# ===============================
# router_node.py
# ===============================

def router_node(state):

    print("✅ router_node executed")

    msg = state["user_message"].lower()

    # -----------------------------
    # PRICE PREDICTION
    # -----------------------------
    if any(word in msg for word in [
        "predict",
        "prediction",
        "estimated price",
        "estimate",
        "price prediction",
        "what should this cost",
        "property value"
    ]):

        state["route"] = "prediction"

    # -----------------------------
    # NEGOTIATION
    # -----------------------------
    elif any(word in msg for word in [
        "negotiate",
        "negotiable",
        "negotiation",
        "discount",
        "best price",
        "reduce price",
        "deal"
    ]):

        state["route"] = "negotiation"

    # -----------------------------
    # VALUATION
    # -----------------------------
    elif any(word in msg for word in [
        "overpriced",
        "undervalued",
        "valuation",
        "worth",
        "fair price"
    ]):

        state["route"] = "valuation"

    # -----------------------------
    # GENERAL CHAT
    # -----------------------------
    else:

        state["route"] = "general"

    # -----------------------------
    # DEBUG ROUTE
    # -----------------------------
    print(f"➡️ Routed to: {state['route']}")

    return state