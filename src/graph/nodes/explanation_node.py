# ===============================
# explanation_node.py
# ===============================

from src.agents.explanation_agent import generate_comparison_explanation

from src.llm.deepseek_client import ask_deepseek_stream


def explanation_node(state):
    print("✅ explanation_node executed")

    explanation = generate_comparison_explanation(
        state["comparison_raw"],
        state["comparison_result"],
        ask_deepseek_stream
    )

    state["explanation"] = explanation

    return state