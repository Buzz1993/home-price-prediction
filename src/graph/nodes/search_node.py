# ===============================
# search_node.py
# ===============================

from src.agents.search_agent import run_search_pipeline


def search_node(state):
    print("✅ search_node executed")

    df = state["df"]
    X_processed = state["X_processed"]

    recs = run_search_pipeline(
        df,
        X_processed,
        state["filters"],
        state["intent"],
        state["slider_weights"],
        state["mode"]
    )

    state["recommendations"] = recs #input and similar properties get stored in state["recommendations"]

    return state