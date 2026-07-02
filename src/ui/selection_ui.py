# ===============================
# selection_ui.py
# ===============================
#These functions in selection_ui.py mainly help create and manage the Streamlit UI for property selection.
#How the property tables look
#How checkboxes work
#How selected properties appear
#How compare/delete actions behave
    

import streamlit as st
import pandas as pd



# =============================
# ADD PROPERTY TO GLOBAL SELECTION
# =============================
def add_to_selected(df_row):
    """
    Add selected property rows to the global
    selected properties dataframe and
    sync them with the active thread.
    """
    if df_row is None or len(df_row) == 0:
        return

    existing = st.session_state.selected_properties

    if existing.empty:
        combined = df_row.copy()
    else:
        combined = pd.concat([existing, df_row]).drop_duplicates(subset=["id"])

    # ✅ GLOBAL STORE
    st.session_state.selected_properties = combined

    # ✅ 🔥 CRITICAL FIX: SYNC WITH ACTIVE THREAD
    active_thread = st.session_state.get("active_thread")
    if active_thread:
        st.session_state.threads[active_thread]["selected"] = combined.copy()


# =============================
# CLEAR SELECTED
# =============================
def clear_selected():
    """
    Reset all selected properties and comparison-related session state.
    """

    # clear selected dataframe
    st.session_state.selected_properties = pd.DataFrame()

    # clear checkbox states
    st.session_state.sim_selected_keys = set()
    st.session_state.input_selected_keys = set()

    # clear thread selected data
    active_thread = st.session_state.get("active_thread")

    if active_thread:
        st.session_state.threads[active_thread]["selected"] = pd.DataFrame()

    # clear comparison state
    if active_thread:
        st.session_state.threads[active_thread]["comparison_result"] = None
        st.session_state.threads[active_thread]["comparison_raw"] = None

    # stop comparison rerun trigger
    st.session_state.run_comparison_now = False


# =============================
# RENDER SELECTED PANEL
# =============================
def render_selected_panel():
    # =============================
    # Display selected properties tray
    # =============================
    """
    Display selected properties in the comparison tray and manage user actions such as compare, delete, and clear.
    """

    st.subheader("📌 Selected Properties (Comparison Tray)")

    col1, col2 = st.columns([8, 1])

    with col2:
        if st.button("🗑️ Clear"):
            clear_selected()

    selected_df = st.session_state.selected_properties.copy()

    if selected_df.empty:
        st.info("No properties selected yet. Use checkboxes to add.")
        return None

    # remove old Select column if exists
    if "Select" in selected_df.columns:
        selected_df = selected_df.drop(columns=["Select"])

    # add control columns
    selected_df.insert(0, "Compare", False)
    selected_df.insert(1, "Delete", False)

    edited_selected = st.data_editor(
        selected_df,
        width="stretch",
        hide_index=True,
        key="selected_editor",
        disabled=["id","city","location","bed","area","costpersqft","furnish","price","builder"]
    )

    # HANDLE DELETE
    rows_to_delete = edited_selected[edited_selected["Delete"] == True]

    if not rows_to_delete.empty:
        ids_to_remove = rows_to_delete["id"]

        st.session_state.selected_properties = (
            st.session_state.selected_properties[
                ~st.session_state.selected_properties["id"].isin(ids_to_remove)
            ]
        )

        print("************selected properties deleted - selection_ui.py************")
        st.rerun()

    # 🔥 DO NOT overwrite compare selection here
    # Only sync tray data separately

    active_thread = st.session_state.get("active_thread")

    if active_thread:

        if "tray_selected" not in st.session_state.threads[active_thread]:
            st.session_state.threads[active_thread]["tray_selected"] = pd.DataFrame()

        st.session_state.threads[active_thread]["tray_selected"] = (
            st.session_state.selected_properties.copy()
        )

    #print("================================")
    #print("edited selected", edited_selected)
    return edited_selected #return dataframe of selected properties with "Compare" and "Delete" columns to track user actions in the comparison tray




def update_selection(selected_keys, edited_df):

    current_selected = set(
        edited_df[edited_df["Select"] == True].index
    )

    added = current_selected - selected_keys
    removed = selected_keys - current_selected

    return current_selected, added, removed


# =============================
# GENERIC SELECTION HANDLER
# =============================
def handle_selection(df, selected_keys_state_key, editor_key):
    """
    Handle property selection and
    synchronize checkbox selection state
    with session storage.
    """
    df = df.copy()

    # restore selection state
    df.insert(0, "Select", df.index.isin(st.session_state[selected_keys_state_key]))

    edited = st.data_editor(
        df,
        width="stretch",
        key=editor_key,
        hide_index=True
    )

    current_selected, added, removed = update_selection(
        st.session_state[selected_keys_state_key],
        edited
    )

    # ADD NEW
    if added:
        new_rows = df.loc[list(added)]
        add_to_selected(new_rows)

        # 🔥 FORCE UI REFRESH (FIX DELAY BUG)
        st.session_state[selected_keys_state_key] = current_selected
        st.rerun()


    # REMOVE
    if removed and not st.session_state.selected_properties.empty:
        ids_to_remove = df[df.index.isin(list(removed))]["id"]

        st.session_state.selected_properties = (
            st.session_state.selected_properties[
                ~st.session_state.selected_properties["id"].isin(ids_to_remove)
            ]
        )

        st.session_state[selected_keys_state_key] = current_selected

        # 🔥 IMPORTANT (YOU MISSED THIS)
        st.rerun()


    # fallback update
    st.session_state[selected_keys_state_key] = current_selected



# =============================
# RENDER INPUT PROPERTIES
# =============================
def render_input_properties(input_df, thread_id):
    """
    Display input property table
    with property selection support.
    """

    st.subheader("🏠 Input Property")

    handle_selection(
        input_df,
        "input_selected_keys",
        f"input_editor_{thread_id}"
    )

    # Show warning AFTER displaying the property
    if st.session_state.get("single_property_warning", False):
        st.warning(
            "Only one property is available for the current selection in Static Mode. "
            "Please try Dynamic Mode for more recommendations."
        )


# =============================
# RENDER SIMILAR PROPERTIES
# =============================
def render_similar_properties(sim_df, thread_id):
    """
    Display similar properties table
    with property selection support.
    """
    if sim_df is None or sim_df.empty:
        return
    
    st.subheader("✨ Similar Properties")

    handle_selection(
        sim_df,
        "sim_selected_keys",
        f"sim_editor_{thread_id}"
    )