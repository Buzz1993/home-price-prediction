# ===============================
# thread_manager.py
# ===============================

import pandas as pd


def create_thread(name="New Chat", data=None):
    """
    Create and return a new thread structure.
    """
    return {
        "messages": [],
        "data": data,
        "selected": pd.DataFrame(),
        "comparison_result": None,
        "comparison_raw": None,
        "auto_compare_explain": False,
        "explanation_done": False,
        "last_explanation": None,
        "show_explanation": False,
        "name": name
    }