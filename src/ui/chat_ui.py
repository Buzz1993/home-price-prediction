# ===============================
# chat_ui.py (FINAL FIXED VERSION)
# ===============================

import streamlit as st

from src.graph.workflow import chat_graph

from src.agents.explanation_agent import generate_comparison_explanation
from src.llm.deepseek_client import ask_deepseek_stream

# =============================
# MAIN CHAT UI
# =============================
def handle_chat(thread, recs, edited_selected=None):

    # =============================
    # 🔥 SHOW CHAT HISTORY
    # =============================
    for role, msg in thread.get("messages", []):
        with st.chat_message("user" if role == "USER" else "assistant"):
            st.markdown(msg)

    # =============================
    # 💬 USER INPUT BOX (NEW)
    # =============================
    user_msg = st.chat_input("Ask anything about properties...")

    if user_msg:

        # -------------------------
        # SHOW USER MESSAGE
        # -------------------------
        with st.chat_message("user"):
            st.markdown(user_msg)

        thread["messages"].append(("USER", user_msg))

        # -------------------------
        # RUN CHAT GRAPH
        # -------------------------
        with st.chat_message("assistant"):

            initial_state = {
                # Actual values are stored and updated inside this runtime state dictionary while the graph executes.

                # -----------------------------
                # REQUIRED CHAT INPUT
                # -----------------------------
                "user_message": user_msg,

                # -----------------------------
                # PROPERTY CONTEXT
                # -----------------------------
                "recommendations": recs,

                "selected_properties": thread.get("selected"),

                "comparison_result": thread.get("comparison_result"),

                "comparison_raw": thread.get("comparison_raw"),

                "explanation": thread.get("last_explanation"),

                # -----------------------------
                # REQUIRED GRAPH FIELDS
                # -----------------------------
                "memory": [],

                "route": "",

                "response": ""
            }

            final_state = chat_graph.invoke(initial_state)

            full_response = final_state["response"]

            st.markdown(full_response)

        # SAVE RESPONSE
        thread["messages"].append(("AI", full_response))