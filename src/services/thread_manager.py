# ===============================
# thread_manager.py
# ===============================

import pandas as pd

#cetralized thread management for creating and updating threads in session state for both when we click on the search button in recommendations.py- 
#and when we click on the New chat button in sidebar.py

# thread is LONG-TERM APP MEMORY stored inside st.session_state.
# It persists across Streamlit reruns and stores permanent UI data
# like messages, recommendations, comparison results, and explanations.
def create_thread(name="New Chat", data=None):
    """
    Creates and returns a new chat thread with default values.
    """
    return {
        "messages": [],
        "memory": [],
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