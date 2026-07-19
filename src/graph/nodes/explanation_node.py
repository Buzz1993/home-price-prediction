# ===============================
# explanation_node.py
# ===============================

from src.agents.explanation_agent import generate_comparison_explanation

from src.llm.provider_router import ask_llm_stream


def explanation_node(state):
    print("✅ explanation_node executed")

    explanation = generate_comparison_explanation(
        state["comparison_raw"],
        state["comparison_result"],
        ask_llm_stream
    )

    state["explanation"] = explanation #save the generated explanation in state["explanation"]

    return state