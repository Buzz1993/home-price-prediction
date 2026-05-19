# ===============================
# comparison_node.py
# ===============================

from src.services.comparison_service import run_comparison


def comparison_node(state):
    print("✅ comparison_node executed")

    selected_df = state["selected_properties"]

    raw_df, compare_df = run_comparison(selected_df)

    state["comparison_raw"] = raw_df
    state["comparison_result"] = compare_df

    return state