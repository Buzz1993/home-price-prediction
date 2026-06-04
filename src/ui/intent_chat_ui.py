# =====================================================================
# src/ui/intent_chat_ui.py (DATA EDITOR CHECKBOX VERSION - WITH TRAY SELECTION)
# =====================================================================

import streamlit as st
import pandas as pd
from src.services.chat_service import parse_intent_and_execute

def render_intent_chat_workspace():
    """
    Renders the standalone conversational BM25 search workspace.
    Uses st.data_editor to embed selection checkboxes directly inside the table.
    """
    # 1. Initialize core state trackers if missing from session memory pools
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "comparison_tray" not in st.session_state:
        st.session_state.comparison_tray = []

    # Layout Configuration: Main Chat Pane alongside an actionable Selection Tray
    main_chat_col, sidebar_tray_col = st.columns([3, 1])

    # -----------------------------------------------------------------
    # LEFT COLUMN: The Intent-Driven Chat Workspace
    # -----------------------------------------------------------------
    with main_chat_col:
        st.title("💬 Real Estate Intent Discovery Assistant")
        st.caption("Ask questions naturally: e.g., 'Want 2bhk property with cctv near goregaon railway station'")
        st.write("---")
        

        #chat histry can be like this 
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

        # Render historical conversation elements sequentially
        for msg_idx, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]): # Render user and assistant messages in chat format
                st.markdown(message["text"]) # Render the text content of the message
                
                # If the message contains search results data payload
                if "data" in message:
                    results_list = message["data"] # data is the list of dictionaries representing search results sent by the assistant message payload as shown in above example
                    df_display = pd.DataFrame(results_list)
                    
                    # If a property is already sitting in sidebar tray and if we run a brand new search, but this search 
                    # brings up that same property again, the checkbox will already be checked the moment the data loads.
                    df_display["Select"] = df_display["id"].apply(
                        lambda x: x in st.session_state.comparison_tray
                    )
                    
                    # Rearrange columns so the selection checkbox stays on the far left
                    display_cols = ["Select", "id", "price", "bhk_type", "location", "amenities_mcp", "search_score"]
                    df_display = df_display[display_cols]
                    
                    st.write("🎯 **Search Results:** Check rows to stage them in your active evaluation tray:")
                    
                    # Creates an interactive table. Returns the updated dataframe after the user makes changes.
                    edited_df = st.data_editor(
                        df_display,
                        key=f"editor_{msg_idx}",
                        hide_index=True,
                        disabled=[c for c in display_cols if c != "Select"], # Lock all columns except 'Select'
                        column_config={
                            "Select": st.column_config.CheckboxColumn(
                                "Select",
                                help="Check to add this property to your active tracking tray",
                                default=False,
                            ),
                            "id": st.column_config.TextColumn("Property ID"),
                            "price": st.column_config.NumberColumn("Price (Cr)", format="₹%.2f"),
                            "bhk_type": st.column_config.TextColumn("BHK"),
                            "location": st.column_config.TextColumn("Locality"),
                            "amenities_mcp": st.column_config.TextColumn("Amenities / Features"),
                            "search_score": st.column_config.NumberColumn("BM25 Score", format="%.4f")
                        },
                        use_container_width=True
                    )
                    
                    # Detect user changes inside the interactive table block
                    # Cross-reference edited_df vs your global tray state to find adjustments
                    for _, row in edited_df.iterrows():
                        p_id = row["id"] # Extract the property ID for this row
                        is_checked = row["Select"] # True if user checked the box, False if unchecked
                        in_tray = p_id in st.session_state.comparison_tray # Check if this property ID is already in the global tray state
                        
                        # Case A (Checking a box):If we click an empty checkbox, the code instantly saves that property's ID into your right-hand sidebar tray and shows a success notification.
                        if is_checked and not in_tray:
                            st.session_state.comparison_tray.append(p_id)
                            st.toast(f"Staged {p_id[:10]}... to tray! 📌")
                            st.rerun()
                        
                        #Case B (Checking a box):If we uncheck a box, the code deletes that property from your sidebar tray and shows a removal notification.
                        elif not is_checked and in_tray:
                            st.session_state.comparison_tray.remove(p_id)
                            st.toast(f"Removed {p_id[:10]}... from tray. 🗑️")
                            st.rerun()
                                
                elif "comparison_data" in message:
                    comp_data = message["comparison_data"]
                    st.success(f"**Winner Selected:** Property {comp_data['winner']['id']} (Score: {comp_data['winner']['overall_score']})")
                    st.info(f"**Verdict Decision:** {comp_data['winner']['verdict']}")
                    st.write(f"**Justification Breakdown:** {comp_data['winner']['comparison_reason']}")
                    st.dataframe(pd.DataFrame(comp_data["rankings"]), hide_index=True)

                elif "rental_data" in message:

                    rental_df = pd.DataFrame(
                        message["rental_data"]
                    )

                    st.dataframe(
                        rental_df,
                        use_container_width=True
                    )


        # Capturing New Conversational Prompt Inputs
        if user_input := st.chat_input("Ask anything about properties..."):
            with st.chat_message("user"):
                st.markdown(user_input)
            
            st.session_state.chat_history.append({"role": "user", "text": user_input})
            
            with st.spinner("Processing node routing matrices..."):
                response_payload = parse_intent_and_execute(user_input, st.session_state.comparison_tray)
                
            with st.chat_message("assistant"):
                if response_payload["type"] == "text":
                    bot_text = response_payload["content"]
                    st.markdown(bot_text)
                    st.session_state.chat_history.append({"role": "assistant", "text": bot_text})
                    
                elif response_payload["type"] == "search_results":
                    bot_text = "🎯 Here are the highest ranking properties matching your intent parameters:"
                    st.markdown(bot_text)
                    
                    results_list = response_payload["content"]
                    
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "text": bot_text, 
                        "data": results_list
                    })
                    st.rerun()
                    
                elif response_payload["type"] == "comparison":
                    bot_text = "🏆 **Investment Analytical Evaluation Completed!**"
                    st.markdown(bot_text)
                    
                    comp_data = response_payload["content"]
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "text": bot_text,
                        "comparison_data": comp_data
                    })
                    st.rerun()

                elif response_payload["type"] == "rental":

                    bot_text = "🏠 Rental Analysis Completed"

                    st.markdown(bot_text)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "text": bot_text,
                        "rental_data": response_payload["content"]
                    })

                    st.rerun()

    # -----------------------------------------------------------------
    # RIGHT COLUMN: Persistent Visual Comparison Tray Tracker (With Action Buttons)
    # -----------------------------------------------------------------
    with sidebar_tray_col:
        st.subheader("📌 Active Comparison Tray")
        st.write("Properties staged for multi-node evaluation:")
        
        # Initialize an active comparison selection list in session state if not present
        if "active_comparison_selection" not in st.session_state:
            st.session_state.active_comparison_selection = []

        if not st.session_state.comparison_tray:
            st.info("Tray is empty. Add properties from chat search items.")
        else:
            # Build data frame containing both operational columns
            tray_df = pd.DataFrame({
                "Compare": [pid in st.session_state.active_comparison_selection for pid in st.session_state.comparison_tray],
                "Delete": [False] * len(st.session_state.comparison_tray),
                "Staged Property ID": st.session_state.comparison_tray
            })
            
            # Render dual-action data editor
            edited_tray_df = st.data_editor(
                tray_df,
                key="sidebar_tray_dual_control",
                hide_index=True,
                disabled=["Staged Property ID"], 
                column_config={
                    "Compare": st.column_config.CheckboxColumn(
                        "Compare",
                        help="Check to select this property for the active comparison execution",
                        default=True
                    ),
                    "Delete": st.column_config.CheckboxColumn(
                        "🗑️",
                        help="Check to completely remove this property from your tray",
                        default=False
                    ),
                    "Staged Property ID": st.column_config.TextColumn("Staged Property ID")
                },
                use_container_width=True
            )
            
            # Process state mutations based on user interaction
            updated_selection = []
            tray_mutated = False
            
            for idx, row in edited_tray_df.iterrows():
                pid = row["Staged Property ID"]
                
                # Action 1: Handle Deletion Priority
                if row["Delete"]:
                    if pid in st.session_state.comparison_tray:
                        st.session_state.comparison_tray.remove(pid)
                    if pid in st.session_state.active_comparison_selection:
                        st.session_state.active_comparison_selection.remove(pid)
                    st.toast(f"Removed {pid[:10]}... from data pool. 🗑️")
                    tray_mutated = True
                else:
                    # Action 2: Maintain Active Comparison Selection State
                    if row["Compare"]:
                        updated_selection.append(pid)

            # Update the active execution selection tracking pool
            st.session_state.active_comparison_selection = updated_selection
            
            if tray_mutated:
                st.rerun()
                
            st.caption(f"📊 *Selected for comparison: {len(st.session_state.active_comparison_selection)} of {len(st.session_state.comparison_tray)} properties*")

            # --- NEW INTERACTIVE ACTION BUTTONS ---
            st.write("") # Spacer
            
            # 1. RUN ANALYTICAL COMPARISON BUTTON
            # Disable the button visually if there aren't at least 2 properties checked
            disable_comparison = len(st.session_state.active_comparison_selection) < 2
            
            if st.button(
                "🏆 Compare Properties", 
                use_container_width=True, 
                type="primary", # Highlights the main action button in Streamlit
                disabled=disable_comparison,
                help="Execute multi-node investment comparison matrix for checked items"
            ):
                # Programmatically append user message to chat history so the workspace feels natural
                trigger_prompt = "Compare selected properties from my active tray"
                st.session_state.chat_history.append({"role": "user", "text": trigger_prompt})
                
                # Run backend analytics pipeline with the checked subsets
                with st.spinner("Processing node routing matrices..."):
                    response_payload = parse_intent_and_execute(
                        "compare properties",  # Forces the chat engine into comparison mode
                        st.session_state.active_comparison_selection
                    )
                
                # Append the response data block back into chat history for rendering
                if response_payload["type"] == "comparison":
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "text": "🏆 **Investment Analytical Evaluation Completed!**",
                        "comparison_data": response_payload["content"]
                    })
                elif response_payload["type"] == "text":
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "text": response_payload["content"]
                    })
                
                st.rerun()

            # 2. CLEAR ENTIRE TRAY BUTTON
            if st.button("🗑️ Clear Entire Tray", use_container_width=True):
                st.session_state.comparison_tray = []
                st.session_state.active_comparison_selection = []
                st.rerun()
                
            st.write("---")
            st.caption("💡 *Tip: Check at least 2 properties under 'Compare' and click the **Compare Properties** button above to invoke your analytical node framework!*")