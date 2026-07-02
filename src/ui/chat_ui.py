# ===============================
# chat_ui.py (FINAL FIXED VERSION)
# ===============================

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

import streamlit as st

from src.graph.workflow import chat_graph

from src.agents.explanation_agent import generate_comparison_explanation
from src.llm.deepseek_client import ask_deepseek_stream

# =============================
# MAIN CHAT UI
# =============================
def handle_chat(thread, recs):

    # =============================
    # 🔥 SHOW CHAT HISTORY
    # =============================
    # Get old/stored chat messages from thread["messages"] and display them in the UI.
    for role, msg in thread.get("messages", []): 
        with st.chat_message("user" if role == "USER" else "assistant"): 
            st.markdown(msg)

    # =============================
    # 💬 USER INPUT BOX (NEW)
    # =============================
    user_msg = st.chat_input("Ask anything about properties...") # New chat input box for user to ask questions 

    if user_msg:

        # -------------------------
        # SHOW USER MESSAGE
        # -------------------------
        # Display the current user message in the UI and store it in thread chat history.
        with st.chat_message("user"):
            st.markdown(user_msg)

        thread["messages"].append(("USER", user_msg)) # Save the new user message to thread["messages"] i.e in thread chat history

        # -------------------------
        # RUN LANGGRAPH
        # -------------------------
        #Send property data, comparison data, explanation data and memory to the graph.
        with st.chat_message("assistant"):

            initial_state = {

                "user_message": user_msg,

                "recommendations": recs,

                "selected_properties": thread.get("selected"),

                "comparison_result": thread.get("comparison_result"),

                "comparison_raw": thread.get("comparison_raw"),

                "explanation": thread.get("last_explanation"),

                # keep previous memory
                "memory": thread.get("memory", []),

                "route": "",

                "response": ""
            }

            # print("========================================")
            # print("\nSELECTED COLUMNS:")
            # print(thread.get("selected").columns.tolist())
            # print(thread.get("selected").shape)

            # print("\nCOMPARISON_RESULT COLUMNS:")
            # print(thread.get("comparison_result").columns.tolist())
            # print(thread.get("comparison_result").shape)

            # print("\nCOMPARISON_RAW COLUMNS:")
            # print(thread.get("comparison_raw").columns.tolist())
            # print(thread.get("comparison_raw").shape)
            # print("========================================")


            final_state = chat_graph.invoke(initial_state)

            # Get the latest memory updated inside memory_node() and save it back into the thread.
            thread["memory"] = final_state.get(
                "memory",
                thread.get("memory", [])
            )

            # GET AI RESPONSE
            full_response = final_state["response"]
            
            # DISPLAY RESPONSE
            st.markdown(full_response)

        # SAVE CHAT HISTORY
        thread["messages"].append(("AI", full_response))