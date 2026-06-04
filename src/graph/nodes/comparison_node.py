# ===============================
# comparison_node.py
# ===============================

from src.services.comparison_service import run_comparison


def comparison_node(state):
    print("==============333===============")
    print("✅ comparison_node executed") 

    selected_df = state["selected_properties"] #selected properties dataframe with only rows where Compare column is True (checkbox selected)

    raw_df, compare_df = run_comparison(selected_df)

    state["comparison_raw"] = raw_df 
    state["comparison_result"] = compare_df 

    print("=" * 50)
    print("comaprison_node selected_df columns:", selected_df.columns.tolist())
    print("=" * 50)

    return state