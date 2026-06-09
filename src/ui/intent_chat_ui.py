# # =====================================================================
# # src/ui/intent_chat_ui.py (DATA EDITOR CHECKBOX VERSION - WITH TRAY SELECTION)
# # =====================================================================

# import streamlit as st
# import pandas as pd
# from src.services.chat_service import parse_intent_and_execute

# def render_intent_chat_workspace():
#     """
#     Renders the standalone conversational BM25 search workspace.
#     Uses st.data_editor to embed selection checkboxes directly inside the table.
#     """
#     # 1. Initialize core state trackers if missing from session memory pools
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []
#     if "comparison_tray" not in st.session_state:
#         st.session_state.comparison_tray = []

#     # Layout Configuration: Main Chat Pane alongside an actionable Selection Tray
#     main_chat_col, sidebar_tray_col = st.columns([3, 1])

#     # -----------------------------------------------------------------
#     # LEFT COLUMN: The Intent-Driven Chat Workspace
#     # -----------------------------------------------------------------
#     with main_chat_col:
#         st.title("💬 Real Estate Intent Discovery Assistant")
#         st.caption("Ask questions naturally: e.g., 'Want 2bhk property with cctv near goregaon railway station'")
#         st.write("---")
        

#         #chat histry can be like this 
#         # {
#         #      "role": "user", 
#         #      "text": "Show me 2BHK options under 2 Cr near Goregaon."
#         # },
#         # {
#         #      "role": "assistant", 
#         #      "text": "Here are the highest ranking properties matching your parameters:",
#         #      "data": [
#         #          {"id": "PROP-101", "price": 1.85, "bhk_type": "2 BHK", "location": "Goregaon East"},
#         #          {"id": "PROP-102", "price": 1.95, "bhk_type": "2 BHK", "location": "Goregaon West"}
#         #      ]
#         # },

#         # msg_idx is the index for that message and message is the actual message content like shown above where 1st role(user) and text is become message 0 
#         # then role(assistant), text and data is message 1 and so on. 

#         # Render historical conversation elements sequentially
#         for msg_idx, message in enumerate(st.session_state.chat_history):
#             with st.chat_message(message["role"]): # Render user and assistant messages in chat format
#                 st.markdown(message["text"]) # Render the text content of the message
                
#                 # If the message contains search results data payload
#                 if "data" in message:
#                     results_list = message["data"] # data is the list of dictionaries representing search results sent by the assistant message payload as shown in above example
#                     df_display = pd.DataFrame(results_list)
                    
#                     # If a property is already sitting in sidebar tray and if we run a brand new search, but this search 
#                     # brings up that same property again, the checkbox will already be checked the moment the data loads.
#                     df_display["Select"] = df_display["id"].apply(
#                         lambda x: x in st.session_state.comparison_tray
#                     )
                    
#                     # Rearrange columns so the selection checkbox stays on the far left
#                     display_cols = ["Select", "id", "price", "bhk_type", "location", "amenities_mcp", "search_score"]
#                     df_display = df_display[display_cols]
                    
#                     st.write("🎯 **Search Results:** Check rows to stage them in your active evaluation tray:")
                    
#                     # Creates an interactive table. Returns the updated dataframe after the user makes changes.
#                     edited_df = st.data_editor(
#                         df_display,
#                         key=f"editor_{msg_idx}",
#                         hide_index=True,
#                         disabled=[c for c in display_cols if c != "Select"], # Lock all columns except 'Select'
#                         column_config={
#                             "Select": st.column_config.CheckboxColumn(
#                                 "Select",
#                                 help="Check to add this property to your active tracking tray",
#                                 default=False,
#                             ),
#                             "id": st.column_config.TextColumn("Property ID"),
#                             "price": st.column_config.NumberColumn("Price (Cr)", format="₹%.2f"),
#                             "bhk_type": st.column_config.TextColumn("BHK"),
#                             "location": st.column_config.TextColumn("Locality"),
#                             "amenities_mcp": st.column_config.TextColumn("Amenities / Features"),
#                             "search_score": st.column_config.NumberColumn("BM25 Score", format="%.4f")
#                         },
#                         use_container_width=True
#                     )
                    
#                     # Detect user changes inside the interactive table block
#                     # Cross-reference edited_df vs your global tray state to find adjustments
#                     for _, row in edited_df.iterrows():
#                         p_id = row["id"] # Extract the property ID for this row
#                         is_checked = row["Select"] # True if user checked the box, False if unchecked
#                         in_tray = p_id in st.session_state.comparison_tray # Check if this property ID is already in the global tray state
                        
#                         # Case A (Checking a box):If we click an empty checkbox, the code instantly saves that property's ID into your right-hand sidebar tray and shows a success notification.
#                         if is_checked and not in_tray:
#                             st.session_state.comparison_tray.append(p_id)
#                             st.toast(f"Staged {p_id[:10]}... to tray! 📌")
#                             st.rerun()
                        
#                         #Case B (Checking a box):If we uncheck a box, the code deletes that property from your sidebar tray and shows a removal notification.
#                         elif not is_checked and in_tray:
#                             st.session_state.comparison_tray.remove(p_id)
#                             st.toast(f"Removed {p_id[:10]}... from tray. 🗑️")
#                             st.rerun()
                                
#                 elif "comparison_data" in message:
#                     comp_data = message["comparison_data"]
#                     st.success(f"**Winner Selected:** Property {comp_data['winner']['id']} (Score: {comp_data['winner']['overall_score']})")
#                     st.info(f"**Verdict Decision:** {comp_data['winner']['verdict']}")
#                     st.write(f"**Justification Breakdown:** {comp_data['winner']['comparison_reason']}")
#                     st.dataframe(pd.DataFrame(comp_data["rankings"]), hide_index=True)

#                 elif "rental_data" in message:

#                     rental_df = pd.DataFrame(
#                         message["rental_data"]
#                     )

#                     st.dataframe(
#                         rental_df,
#                         use_container_width=True
#                     )


#         # Capturing New Conversational Prompt Inputs
#         if user_input := st.chat_input("Ask anything about properties..."):
#             with st.chat_message("user"):
#                 st.markdown(user_input)
            
#             st.session_state.chat_history.append({"role": "user", "text": user_input})
            
#             with st.spinner("Processing node routing matrices..."):
#                 response_payload = parse_intent_and_execute(user_input, st.session_state.comparison_tray)
                
#             with st.chat_message("assistant"):
#                 if response_payload["type"] == "text":
#                     bot_text = response_payload["content"]
#                     st.markdown(bot_text)
#                     st.session_state.chat_history.append({"role": "assistant", "text": bot_text})
                    
#                 elif response_payload["type"] == "search_results":
#                     bot_text = "🎯 Here are the highest ranking properties matching your intent parameters:"
#                     st.markdown(bot_text)
                    
#                     results_list = response_payload["content"]
                    
#                     st.session_state.chat_history.append({
#                         "role": "assistant", 
#                         "text": bot_text, 
#                         "data": results_list
#                     })
#                     st.rerun()
                    
#                 elif response_payload["type"] == "comparison":
#                     bot_text = "🏆 **Investment Analytical Evaluation Completed!**"
#                     st.markdown(bot_text)
                    
#                     comp_data = response_payload["content"]
#                     st.session_state.chat_history.append({
#                         "role": "assistant",
#                         "text": bot_text,
#                         "comparison_data": comp_data
#                     })
#                     st.rerun()

#                 elif response_payload["type"] == "rental":

#                     bot_text = "🏠 Rental Analysis Completed"

#                     st.markdown(bot_text)

#                     st.session_state.chat_history.append({
#                         "role": "assistant",
#                         "text": bot_text,
#                         "rental_data": response_payload["content"]
#                     })

#                     st.rerun()

#     # -----------------------------------------------------------------
#     # RIGHT COLUMN: Persistent Visual Comparison Tray Tracker (With Action Buttons)
#     # -----------------------------------------------------------------
#     with sidebar_tray_col:
#         st.subheader("📌 Active Comparison Tray")
#         st.write("Properties staged for multi-node evaluation:")
        
#         # Initialize an active comparison selection list in session state if not present
#         if "active_comparison_selection" not in st.session_state:
#             st.session_state.active_comparison_selection = []

#         if not st.session_state.comparison_tray:
#             st.info("Tray is empty. Add properties from chat search items.")
#         else:
#             # Build data frame containing both operational columns
#             tray_df = pd.DataFrame({
#                 "Compare": [pid in st.session_state.active_comparison_selection for pid in st.session_state.comparison_tray],
#                 "Delete": [False] * len(st.session_state.comparison_tray),
#                 "Staged Property ID": st.session_state.comparison_tray
#             })
            
#             # Render dual-action data editor
#             edited_tray_df = st.data_editor(
#                 tray_df,
#                 key="sidebar_tray_dual_control",
#                 hide_index=True,
#                 disabled=["Staged Property ID"], 
#                 column_config={
#                     "Compare": st.column_config.CheckboxColumn(
#                         "Compare",
#                         help="Check to select this property for the active comparison execution",
#                         default=True
#                     ),
#                     "Delete": st.column_config.CheckboxColumn(
#                         "🗑️",
#                         help="Check to completely remove this property from your tray",
#                         default=False
#                     ),
#                     "Staged Property ID": st.column_config.TextColumn("Staged Property ID")
#                 },
#                 use_container_width=True
#             )
            
#             # Process state mutations based on user interaction
#             updated_selection = []
#             tray_mutated = False
            
#             for idx, row in edited_tray_df.iterrows():
#                 pid = row["Staged Property ID"]
                
#                 # Action 1: Handle Deletion Priority
#                 if row["Delete"]:
#                     if pid in st.session_state.comparison_tray:
#                         st.session_state.comparison_tray.remove(pid)
#                     if pid in st.session_state.active_comparison_selection:
#                         st.session_state.active_comparison_selection.remove(pid)
#                     st.toast(f"Removed {pid[:10]}... from data pool. 🗑️")
#                     tray_mutated = True
#                 else:
#                     # Action 2: Maintain Active Comparison Selection State
#                     if row["Compare"]:
#                         updated_selection.append(pid)

#             # Update the active execution selection tracking pool
#             st.session_state.active_comparison_selection = updated_selection
            
#             if tray_mutated:
#                 st.rerun()
                
#             st.caption(f"📊 *Selected for comparison: {len(st.session_state.active_comparison_selection)} of {len(st.session_state.comparison_tray)} properties*")

#             # --- NEW INTERACTIVE ACTION BUTTONS ---
#             st.write("") # Spacer
            
#             # 1. RUN ANALYTICAL COMPARISON BUTTON
#             # Disable the button visually if there aren't at least 2 properties checked
#             disable_comparison = len(st.session_state.active_comparison_selection) < 2
            
#             if st.button(
#                 "🏆 Compare Properties", 
#                 use_container_width=True, 
#                 type="primary", # Highlights the main action button in Streamlit
#                 disabled=disable_comparison,
#                 help="Execute multi-node investment comparison matrix for checked items"
#             ):
#                 # Programmatically append user message to chat history so the workspace feels natural
#                 trigger_prompt = "Compare selected properties from my active tray"
#                 st.session_state.chat_history.append({"role": "user", "text": trigger_prompt})
                
#                 # Run backend analytics pipeline with the checked subsets
#                 with st.spinner("Processing node routing matrices..."):
#                     response_payload = parse_intent_and_execute(
#                         "compare properties",  # Forces the chat engine into comparison mode
#                         st.session_state.active_comparison_selection
#                     )
                
#                 # Append the response data block back into chat history for rendering
#                 if response_payload["type"] == "comparison":
#                     st.session_state.chat_history.append({
#                         "role": "assistant",
#                         "text": "🏆 **Investment Analytical Evaluation Completed!**",
#                         "comparison_data": response_payload["content"]
#                     })
#                 elif response_payload["type"] == "text":
#                     st.session_state.chat_history.append({
#                         "role": "assistant",
#                         "text": response_payload["content"]
#                     })
                
#                 st.rerun()

#             # 2. CLEAR ENTIRE TRAY BUTTON
#             if st.button("🗑️ Clear Entire Tray", use_container_width=True):
#                 st.session_state.comparison_tray = []
#                 st.session_state.active_comparison_selection = []
#                 st.rerun()
                
#             st.write("---")
#             st.caption("💡 *Tip: Check at least 2 properties under 'Compare' and click the **Compare Properties** button above to invoke your analytical node framework!*")

#==========================================================================================================================================================================
#above code is simply converted into functions as below 
# =====================================================================
# intent_chat_ui.py (FINAL MASTER ARTIFACT - CLEAN & PYTHONIC)
# =====================================================================

import streamlit as st
import pandas as pd
from src.services.chat_service import parse_intent_and_execute

# ---------------------------------------------------------------------
# UI CONFIGURATION & DISPLAY CONSTANTS
# ---------------------------------------------------------------------
SEARCH_DISPLAY_COLUMNS = ["Select", "id", "price", "bhk_type", "location", "search_score"]

RESPONSE_CONFIG = {
    "text": (None, None),
    "search_results": ("🎯 Here are the highest ranking properties matching your intent parameters:", "data"),
    "comparison": ("🏆 **Investment Analytical Evaluation Completed!**", "comparison_data"),
    "rental": ("💰 **Rental Performance Matrix Extractions Completed:**", "rental_data"),
    "prediction": ("🔮 **FastAPI Core Predictive Yield Forecast Completed:**", "prediction_data"),
    "negotiation": ("🤝 **Strategic Buyer Leverage & Talking Points Compiled:**", "negotiation_data"),
    "valuation": ("📊 **Benchmark Deviation Pricing Assessments Completed:**", "valuation_data"),
    "advisor": ("🧠 **High-Conviction Suitability Investment Advice Compiled:**", "advisor_data")
}

SEARCH_COLUMN_CONFIG = {
    "Select": st.column_config.CheckboxColumn("Select", help="Check to stage property", default=False),
    "id": st.column_config.TextColumn("Property ID"),
    "price": st.column_config.NumberColumn("Price (Cr)", format="₹%.2f"),
    "bhk_type": st.column_config.TextColumn("BHK"),
    "location": st.column_config.TextColumn("Locality"),
    "amenities_mcp": st.column_config.TextColumn("Amenities / Features"),
    "search_score": st.column_config.NumberColumn("BM25 Score", format="%.4f")
}

TRAY_COLUMN_CONFIG = {
    "Compare": st.column_config.CheckboxColumn("Compare", help="Select for comparison matrix", default=True),
    "Delete": st.column_config.CheckboxColumn("🗑️", help="Completely remove item", default=False),
    "Staged Property ID": st.column_config.TextColumn("Staged Property ID")
}


# ---------------------------------------------------------------------
# 1. STATE INITIALIZATION & UTILITIES
# ---------------------------------------------------------------------

def init_session_states():
    """Initializes default lists in background session memory."""
    defaults = {
        "chat_history": [],
        "comparison_tray": [],
        "active_comparison_selection": []
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def append_to_chat(role, text, payload_key=None, payload_data=None):
    """Saves a single message entry into the background chat history list."""
    msg = {"role": role, "text": text}
    if payload_key and payload_data is not None:
        msg[payload_key] = payload_data
    st.session_state.chat_history.append(msg)


def process_response(response):
    """Formats and saves the LLM response to the chat history."""
    
    # 1. Grab the response type (like 'text' or 'search_results') and its core content data
    response_type = response["type"]
    content = response["content"]

    # 2. Look up the response type in our configuration dictionary to get its title and key name
    title, payload_key = RESPONSE_CONFIG[response_type]

    # 3. If it has a payload_key, it means this reply contains a data table or structured result
    if payload_key: #payload key are shown in RESPONSE_CONFIG 
        # Save it to chat history with its specific title header and data payload
        append_to_chat("assistant", title, payload_key, content)
    else:
        # Otherwise, it is just a simple text response, so save only the content text
        append_to_chat("assistant", content)


# ---------------------------------------------------------------------
# 2. DATAFRAME BUILDERS & GLOBAL RENDERERS
# ---------------------------------------------------------------------

def build_tray_dataframe():
    """Creates a table showing the properties currently saved in the tray."""
    return pd.DataFrame({
        "Compare": [
            pid in st.session_state.active_comparison_selection
            for pid in st.session_state.comparison_tray
        ],
        "Delete": [False] * len(st.session_state.comparison_tray), # Default state for deletion ticks is blank
        "Staged Property ID": st.session_state.comparison_tray
    })


def build_search_dataframe(results_list):
    """Creates a table for search results synced with active tray selections."""
    # Converts raw search data into a table
    df = pd.DataFrame(results_list) 
    
    # Shows a TICK if the property is already in the tray, and NO TICK if it isn't
    df["Select"] = df["id"].apply( 
        lambda x: x in st.session_state.comparison_tray  #this compariosn_tray is created in session state already in init_session_states function
    )
    return df


def render_dataframe(data):
    """Displays a standardized data table on the screen."""
    st.dataframe(
        pd.DataFrame(data),
        hide_index=True,
        use_container_width=True
    )


# ---------------------------------------------------------------------
# 3. INTERACTIVE WIDGET RENDERERS
# ---------------------------------------------------------------------

def handle_search_result_selection(edited_df):
    """Adds or removes properties from the tray based on user checkbox clicks."""
    mutated = False
    # Loop through the table data received to see what the user clicked
    for _, row in edited_df.iterrows():
        p_id = row["id"]
        is_checked = row["Select"]
        in_tray = p_id in st.session_state.comparison_tray

        # If user selects the box, add it to the comparison_tray
        if is_checked and not in_tray:
            st.session_state.comparison_tray.append(p_id)
            st.toast(f"Staged {p_id[:10]}... to tray! 📌")
            mutated = True

        # Else if user unselects the box, remove it from the comparison_tray
        elif not is_checked and in_tray:
            st.session_state.comparison_tray.remove(p_id)
            st.toast(f"Removed {p_id[:10]}... from tray. 🗑️")
            mutated = True

    # If any selection changed, refresh the screen to update the layout
    if mutated:
        st.rerun()


def render_search_results_table(results_list, msg_idx):
    """Displays search results in an interactive table with selection checkboxes."""
    df_raw = build_search_dataframe(results_list) # Converts raw search data into a table and shows a TICK if the property is already in the tray, and NO TICK if it isn't
    df_display = df_raw[SEARCH_DISPLAY_COLUMNS] # Rearrange columns so the selection checkbox stays on the far left

    st.write("🎯 **Search Results:** Check rows to stage them in your active evaluation tray:")
    
    # Creates an interactive table. Returns the updated dataframe after the user makes changes.
    edited_df = st.data_editor( 
        df_display,
        key=f"editor_{msg_idx}",
        hide_index=True,
        disabled=[c for c in SEARCH_DISPLAY_COLUMNS if c != "Select"],
        column_config=SEARCH_COLUMN_CONFIG,
        use_container_width=True
    )
    handle_search_result_selection(edited_df) # Detect user changes inside the interactive table block and update the tray state accordingly


def render_comparison_result(comp_data):
    """Displays the analytical comparison results, winner info, and ranking tables."""
    
    # 1. Show a green success box highlighting the winning property ID and its final score
    st.success(f"**Winner Selected:** Property {comp_data['winner']['id']} (Score: {comp_data['winner']['overall_score']})")
    
    # 2. Show a blue info box displaying the clear verdict decision from the AI
    st.info(f"**Verdict Decision:** {comp_data['winner']['verdict']}")
    
    # 3. Write out a paragraph breakdown explaining exactly why that property won
    st.write(f"**Justification Breakdown:** {comp_data['winner']['comparison_reason']}")
    
    # 4. Display a structured table comparing the final rankings of all competing properties side-by-side
    render_dataframe(comp_data["rankings"])


def render_extended_data_grids(message):
    """Display prediction, negotiation, valuation, and advisor results in Streamlit UI."""
    if "prediction_data" in message:
        pred_df = pd.DataFrame(message["prediction_data"])
        st.dataframe(pred_df.style.format({"original_price": "₹{:.2f} Cr", "predicted_price": "₹{:.2f} Cr", "margin_diff": "₹{:.2f} Cr"}), hide_index=True, use_container_width=True)
        
    elif "negotiation_data" in message:
        for idx, item in enumerate(message["negotiation_data"]):
            with st.expander(f"🤝 Strategy Guide - Property: {item.get('id')}"):
                st.metric("Target Buying Price", f"₹{item.get('target_price')} Cr", help=f"Suggested discount: {item.get('suggested_discount_percent')}")
                st.markdown(f"**Negotiation Leverage Power:** `{item.get('negotiation_power')}`")
                st.markdown(f"**Strategic Approach:** {item.get('strategy')}")
                st.info(f"**Talking Points:** {item.get('talking_points')}")
                
    elif "valuation_data" in message:
        val_df = pd.DataFrame(message["valuation_data"])
        st.dataframe(val_df[["id", "project_name", "price", "costpersqft", "analysis_msg", "analysis_severity"]], hide_index=True, use_container_width=True)
        
    elif "advisor_data" in message:
        for item in message["advisor_data"]:
            # FIX: Dropped the broken hasattr get_container hack logic completely
            with st.container():
                st.markdown(f"### 🏠 Asset Allocation Analysis - ID: {item.get('id')}")
                st.warning(f"**Investment Suitability:** {item.get('suitable_for')} | **Verdict Status:** {item.get('verdict')}")
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Identified Positives:**\n{item.get('positives')}")
                with col2:
                    st.error(f"**Identified Systemic Risks:**\n{item.get('risks')}")
                st.write("---")
# ---------------------------------------------------------------------
# 4. CHAT STREAM WORKFLOW MANAGERS
# ---------------------------------------------------------------------

#chat history can be like this 
    # {
    #      "role": "user", 
    #      "text": "Show me 2BHK options under 2 Cr near Goregaon."
    # },
    # {
    #      "role": "assistant", 
    #      "text": "Here are the highest ranking properties matching your parameters:",
    #      "data": [
    #          {"id": "PROP-101", "price": 1.85, "bhk_type": "2 BHK", "location": "Goregaon East"},
    #          {"id": "PROP-102", "price": 1.95, "bhk_type": "2 BHK", "location": "Goregaon West"}
    #      ]
    # },

    # msg_idx is the index for that message and message is the actual message content like shown above where 1st role(user) and text is become message 0 
    # then role(assistant), text and data is message 1 and so on.

def render_chat_history():
    """Loops through and renders standard history, updating internal component references safely."""
    for msg_idx, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            # Render the header or conversational message body text first
            st.markdown(message["text"])
            
            # --- STRUCTURED DATA PAYLOAD ROUTING LAYERS ---
            
            # 1. Standard Search Results with Interactive Selection Checkboxes
            if "data" in message: 
                render_search_results_table(message["data"], msg_idx) # get a table of search results with checkboxes
                
            # 2. Multi-Node Stacking Comparison Result
            elif "comparison_data" in message:
                render_comparison_result(message["comparison_data"]) # get a formatted comparison result with winner details and ranking table
                
            # 3. Micro-Rental Performance Dataframe
            elif "rental_data" in message:
                render_dataframe(message["rental_data"]) # get a simple dataframe showing rental performance metrics for the properties under analysis
                
            # 4. Deep Analytical Custom Grid Frameworks
            elif any(k in message for k in ["prediction_data", "negotiation_data", "valuation_data", "advisor_data"]): 
                render_extended_data_grids(message) 
                
            # 5. Fallback Safety Gate
            else:
                pass


def handle_new_chat_input():
    """Handles the user's input, gets the LLM response, and updates the chat."""
    
    # 1. Listen for when the user types something in the chat box and presses Enter
    if user_input := st.chat_input("Ask anything about properties..."):
        
        # 2. Immediately show the user's typed message on the screen
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # 3. Save the user's message into our background chat history list
        append_to_chat("user", user_input)
        
        # 4. Show a loading spinner while the backend database searches for properties
        with st.spinner("Processing node routing matrices..."): 
            response = parse_intent_and_execute(user_input, st.session_state.comparison_tray)
            
        # 5. Process and display the assistant (LLM) response.
        with st.chat_message("assistant"):
            process_response(response)  # Formats data (shows text, search tables, or comparison grids)
            st.rerun()                  # Refresh the screen to cleanly show the updated conversation


# ---------------------------------------------------------------------
# 5. PERSISTENT EVALUATION SIDEBAR TRAYS
# ---------------------------------------------------------------------

def run_comparison():
    """Triggers the property comparison analysis and saves the results to chat."""
    append_to_chat("user", "Compare selected properties from my active tray")
    
    with st.spinner("Processing node routing matrices..."):
        response = parse_intent_and_execute("compare properties", st.session_state.active_comparison_selection)
        
    process_response(response)
    st.rerun()


def handle_tray_mutations(edited_tray_df):
    """Processes deletions and updates selected properties in the comparison tray."""
    updated_selection = []
    tray_mutated = False
    
    for _, row in edited_tray_df.iterrows():
        pid = row["Staged Property ID"]
        
        if row["Delete"]:
            if pid in st.session_state.comparison_tray:
                st.session_state.comparison_tray.remove(pid)
            if pid in st.session_state.active_comparison_selection:
                st.session_state.active_comparison_selection.remove(pid)
            st.toast(f"Removed {pid[:10]}... from data pool. 🗑️")
            tray_mutated = True
        else:
            if row["Compare"]:
                updated_selection.append(pid)

    st.session_state.active_comparison_selection = updated_selection
    if tray_mutated:
        st.rerun()


def render_tray_actions():
    """Renders the buttons to compare properties or clear the tray."""
    disable_comparison = len(st.session_state.active_comparison_selection) < 2
    
    if st.button("🏆 Compare Properties", use_container_width=True, type="primary", disabled=disable_comparison):
        run_comparison()

    if st.button("🗑️ Clear Entire Tray", use_container_width=True):
        st.session_state.comparison_tray = []
        st.session_state.active_comparison_selection = []
        st.rerun()


def render_tray_column():
    """Displays and manages the property comparison sidebar."""
    st.subheader("📌 Active Comparison Tray")
    st.write("Properties staged for multi-node evaluation:")

    if not st.session_state.comparison_tray:
        st.info("Tray is empty. Add properties from chat search items.")
        return

    tray_df = build_tray_dataframe()
    
    edited_tray_df = st.data_editor(
        tray_df,
        key="sidebar_tray_dual_control",
        hide_index=True,
        disabled=["Staged Property ID"],
        column_config=TRAY_COLUMN_CONFIG,
        use_container_width=True
    )
    
    handle_tray_mutations(edited_tray_df)
    
    st.caption(f"📊 *Selected for comparison: {len(st.session_state.active_comparison_selection)} of {len(st.session_state.comparison_tray)} properties*")
    st.write("") 
    
    render_tray_actions()
    
    st.write("---")
    st.caption("💡 *Tip: Check at least 2 properties under 'Compare' and click the **Compare Properties** button above to invoke your analytical node framework!*")


# ---------------------------------------------------------------------
# 6. MASTER DESIGN ORCHESTRATOR 
# ---------------------------------------------------------------------

def render_intent_chat_workspace():
    """Splits the screen into a main chat window and a property sidebar."""
    init_session_states()
    
    main_chat_col, sidebar_tray_col = st.columns([3, 1])
    
    with main_chat_col:
        st.title("🧠 EstateMind Copilot")
        st.caption("Ask questions naturally: e.g., 'Want 2bhk property with cctv near goregaon railway station'")
        st.write("---")
        
        render_chat_history()
        handle_new_chat_input()
        
    with sidebar_tray_col:
        render_tray_column()

