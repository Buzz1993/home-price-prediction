# ===============================
# recommendations.py (REFACTORED)
# ===============================

from concurrent.futures import thread

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import uuid

# ===============================
# IMPORTS
# ===============================
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.data.content_based_filtering import train
from src.ui.sidebar import get_user_intent_and_weights
from src.ui.sidebar import render_thread_sidebar

from src.ui.selection_ui import (
    render_selected_panel,
    render_input_properties,
    render_similar_properties
)

from src.ui.chat_ui import handle_chat
from src.ui.comparison_ui import render_comparison

from src.graph.workflow import search_graph, comparison_graph
from src.services.thread_manager import create_thread

# ===============================
# MEMORY / GLOBAL CONFIG
# ===============================
USER_ID = "default_user"


# =============================
# LOAD SYSTEM DATA
# =============================
@st.cache_resource
def load_system():
    # =============================
    # Load dataset + train model
    # =============================
    """
    Loads dataset and trains recommendation pipeline.
    Returns raw dataframe and processed feature matrix.
    """
    df = pd.read_csv(ROOT_DIR / "data" / "cleaned" / "final_cleaned_rec_data.csv")
    pipe, X = train(df)
    return df, X # returning both the original dataframe(df) and the processed feature matrix(X) after training the model


# =============================
# SESSION STATE INITIALIZATION
# =============================
def init_session_state():
    # =============================
    # Initialize all session variables
    # =============================
    """
    This function is used in Streamlit to make sure all required variables exist inside st.session_state.
    """
    # THREAD SYSTEM
    if "threads" not in st.session_state:  #Creates a dictionary to store: different chat/search threads
        st.session_state.threads = {}

    if "active_thread" not in st.session_state:  #Tracks:which thread is currently active
        st.session_state.active_thread = None

    if "pinned_threads" not in st.session_state:  # Stores IDs of pinned threads to keep them at the top of sidebar
        st.session_state.pinned_threads = []

    # PROPERTY SELECTION
    if "selected_properties" not in st.session_state:  #Stores selected properties in a table format using pandas
        st.session_state.selected_properties = pd.DataFrame()

    if "input_selected_keys" not in st.session_state:  #Tracks selected items from input/search results
        st.session_state.input_selected_keys = set()

    if "sim_selected_keys" not in st.session_state:  #Tracks selected items from similar/recommended properties
        st.session_state.sim_selected_keys = set()

    if "last_changed" not in st.session_state:  # Tracks which filter was last modified (used to trigger updates)
        st.session_state.last_changed = None


# =============================
# FILTER LOGIC
# =============================
def get_filtered_df(df, filters, exclude_col=None):
    # =============================
    # Apply filters dynamically
    # =============================
    """
    Returns filtered dataframe based on user filters.
    """
    temp = df.copy()

    for k, v in filters.items():
        if k == exclude_col:
            continue
        if v != "Any":
            if k in ["builder", "transportation_hubs_clean"]:
                temp = temp[temp[k].str.contains(str(v), case=False, na=False)]
            else:
                temp = temp[temp[k] == v]

    return temp


def get_options(df, col, filters):
    # =============================
    # Get dropdown options
    # =============================
    """
    Returns valid dropdown options based on filtered data.
    """
    temp = get_filtered_df(df, filters, exclude_col=col)
    return ["Any"] + sorted(temp[col].dropna().unique().tolist())


# =============================
# RESET FILTERS
# =============================
def reset_filters(default_filters):
    # =============================
    # Reset all filters to default
    # =============================
    """
    Resets all filter values in session state.
    """
    for k in default_filters:
        st.session_state[k] = "Any"


# =============================
# SEARCH HANDLER
# =============================
def handle_search(df, X_processed, filters, intent, slider_weights, mode):
    # =============================
    # Run search + create/update thread
    # =============================
    """
    Executes search pipeline.
    
    RULE:
    - If active thread is "New Chat" → reuse same thread
    - Else → create new thread
    """

    initial_state = {
        # Actual values are stored and updated inside this runtime state dictionary while the graph executes.

        "df": df,
        "X_processed": X_processed,

        "filters": filters,
        "intent": intent,
        "slider_weights": slider_weights,
        "mode": mode,

        "selected_properties": pd.DataFrame(),

        "recommendations": None,

        "comparison_raw": None,
        "comparison_result": None,

        "explanation": None
    }

    final_state = search_graph.invoke(initial_state)

    recs = final_state["recommendations"] #recs is dictionary having two keys "input" and "similar" and values are input properties and similar properties respectively
    # state["recommendations"] = recs (from search_node.py) stores the recommendation output inside the LangGraph runtime state, where recs is a dictionary containing both-
    # the input property dataframe and the similar properties dataframe. After the graph execution completes, the updated runtime state is returned as final_state, so the-
    # same stored recommendations can later be accessed using recs = final_state["recommendations"].(from recommendations.py)

    #where, similar properties dataframe has additional columns 
    #analysis_flag, analysis_msg, analysis_severity (from analysis agent results)
    #risk_categories, risk_score, risk_label (from risk agent results)
    #future_signals, infra_detected, growth_label, growth_reason, growth_score (from future agent results)
    #negotiation_power, suggested_discount_percent, target_price, price_position, strategy, talking_points (from negotiation agent results)
    #monthly_rent_estimate, annual_rent, rental_yield_percent, demand_level, investment_rating, rental_strategy (from rental agent results)
    #price_score, area_score, amenities_score, location_score, connectivity_score, distance_score, weighted_score,why_recommended,hybrid_score (from hybrid ranking calculations) i.e from hybrid_recommender.py
    #min_rent, max_rent (from search agent))
    #print("=============================")
    # print(st.session_state.threads) # get empty dictionary {} when we run the search(after clicking on search button) for the first time as no thread is created yet


    print("==============111===============")
    #print(recs)
    #print(recs["input"].columns.tolist())
    #print(recs["similar"].columns.tolist())

    print("shape of input property columns", len(recs["input"].columns))
    print("shape of similar property columns", len(recs["similar"].columns))

    if not recs:
        return

    name = f"{st.session_state.city} | {st.session_state.bed} BHK" #given the thread name eg: Mumbai | 2 BHK based on selected city and bed filters
    #print(name)

    active_thread = st.session_state.get("active_thread") #active thread id example:f7c11cb2

    # =========================================
    # ✅ REUSE "NEW CHAT" THREAD
    # =========================================
    # After clicking the "New Chat" button in the UI,
    # a new chat thread gets opened, and for that thread we use the below code.
    if (
        active_thread
        and active_thread in st.session_state.threads
        and st.session_state.threads[active_thread]["name"] == "New Chat"
    ):

        thread = st.session_state.threads[active_thread]

        thread["data"] = recs
        thread["name"] = name

        # optional reset
        thread["comparison_result"] = None
        thread["comparison_raw"] = None
        thread["auto_compare_explain"] = False
        thread["explanation_done"] = False

    # =========================================
    # ✅ CREATE NEW THREAD (NORMAL CASE)
    # =========================================
    else:
        #user already has an existing search thread and now performs another search, So instead of overwriting old thread data, creates a brand new thread
        tid = str(uuid.uuid4())[:8] #generate short unique thread ID example: 'a1b2c3d4' for new search thread
        
        #In threads dict - data get added as in search node.py we store the search results in state["recommendations"] = recs 
        st.session_state.threads[tid] = create_thread( #create new thread in session state with key as thread id and value as dictionary having keys "messages","data","selected","comparison_result","comparison_raw","auto_compare_explain","explanation_done","name" to store all relevant data for that thread
            name=name,
            data=recs
        )

        st.session_state.active_thread = tid
        #print("=============================")
        #print(st.session_state.threads) #On first execution/search, almost everything is empty/default except "data" because only recommendation results exist at that moment.

    st.rerun()


# =============================
# MAIN APP
# =============================
def main():
    # =============================
    # Main Streamlit application
    # =============================
    """
    Runs full property recommendation UI and logic.
    """

    st.set_page_config(layout="wide")
    st.title("🏠 Property Recommendation System")

    df, X_processed = load_system()
    init_session_state()

    # print("\n===== SESSION STATE DEBUG =====")

    # print("threads:")
    # print(st.session_state.threads)

    # print("\nactive_thread:")
    # print(st.session_state.active_thread)

    # print("\npinned_threads:")
    # print(st.session_state.pinned_threads)

    # print("\nselected_properties:")
    # print(st.session_state.selected_properties)

    # print("\ninput_selected_keys:")
    # print(st.session_state.input_selected_keys)

    # print("\nsim_selected_keys:")
    # print(st.session_state.sim_selected_keys)

    # print("\nlast_changed:")
    # print(st.session_state.last_changed)

    # print("================================\n")

    # -----------------------------
    # SIDEBAR
    # -----------------------------
    render_thread_sidebar()   #is responsible for creating and managing the entire sidebar thread UI.

    # -----------------------------
    # 🎯 USER INTENT + SLIDERS
    # -----------------------------
    intent, slider_weights = get_user_intent_and_weights()  # Call the function get_user_intent_and_weights() and store its returned values into: intent and slider_weights

    # -----------------------------
    # FILTER UI
    # -----------------------------
    # Filter helper functions used to initialize state, render UI, and reset filter values.
    from src.ui.filters import (
        get_default_filters,  # Provides default values for all filters.
        init_filter_state,  # If filter does not exist in memory, create it with default value    
        render_filter_ui,  # Renders the top filter/input layer of the UI with static/dynamic mode and dropdown filters for city, location, bed, furnish, price range, transportation hubs, and builder selection.
        reset_filters  # resets filter state values back to defaults
    )

    default_filters = get_default_filters() 
    init_filter_state(default_filters)

    filters, mode = render_filter_ui(df, default_filters)

    # User changed filters/preferences. So current AI-generated property insights/comparison explanation gets reset.
    if "last_changed" in st.session_state and st.session_state.last_changed:
        if st.session_state.active_thread:
            thread = st.session_state.threads[st.session_state.active_thread]
            thread["explanation_done"] = False
            thread["last_explanation"] = None

    st.button("🔄 Reset", on_click=lambda: reset_filters(default_filters)) ## Reset all filters back to their default values when user clicks the Reset button.

    # -----------------------------
    # SEARCH BUTTON
    # -----------------------------
    if st.button("🔍 Search"):

        with st.spinner("🔍 Finding best properties..."):
            handle_search(
                df,
                X_processed,
                filters,
                intent,
                slider_weights,
                mode
            )

    # -----------------------------
    # THREAD DISPLAY + CHAT (FIXED)
    # -----------------------------
    #print("==============222===============")
    if st.session_state.active_thread: # Streamlit session_state is used to store values across app reruns so this if st.session_state.active_thread means - Checks whether an active thread currently exists
    #active thread id example:f7c11cb2
        
        # FETCHES / ACCESSES the currently active thread data from memory.
        # Example: thread["data"], thread["selected"], thread["comparison_result"], thread["messages"], thread["comparison_raw"], thread["auto_compare_explain"], thread["explanation_done"], thread["name"]
        #all this are saved at this step : # ✅ CREATE NEW THREAD (NORMAL CASE)
        thread = st.session_state.threads[st.session_state.active_thread]
        #print("=============================")
        #print("thread data printed: ",thread.keys()) #dict_keys(['messages', 'data', 'selected', 'comparison_result', 'comparison_raw', 'auto_compare_explain', 'explanation_done', 'name'])
        #print(thread.keys() and thread.values()) 


        # Stores recommendation results generated by the search pipeline including:input property dataframe and similar properties dataframe
        # Example:recs["input"], recs["similar"]
        recs = thread["data"] #contain input_df and similar_df
        edited_selected = None

        if recs:
            
            #get selected properties from - Selected Properties (Comparison Tray)
            edited_selected = render_selected_panel()

            selected_for_compare = pd.DataFrame()

            if (
                edited_selected is not None
                and not edited_selected.empty
                and "Compare" in edited_selected.columns
            ):

                selected_for_compare = edited_selected[ #selected_for_compare is a dataframe having only rows where Compare column is True (checkbox selected) from the edited_selected dataframe which is the output of render_selected_panel() function and has all the properties that are currently present in the Selected Properties (Comparison Tray) along with the "Compare" column which is a boolean column created in render_selected_panel() function to track which properties are selected for comparison by the user using checkboxes in the UI.
                    edited_selected["Compare"] == True
                ]

            #store active thread id 
            thread_id = st.session_state.active_thread #get active thread id example:f7c11cb2

            render_input_properties(recs["input"], thread_id) #display input property details in a table
            render_similar_properties(recs["similar"], thread_id) #display similar properties details in a table

            # -----------------------------
            # COMPARE BUTTON
            # -----------------------------
            if len(selected_for_compare) >= 2:

                if st.button("⚖️ Compare Selected Properties"):

                    active_thread = st.session_state.active_thread 
                    thread = st.session_state.threads[active_thread] # get current active thread data example: thread["data"], thread["selected"]
                    # print("==============================")
                    # print("data", thread.get("data")) #recommendation results including input_df and similar_df
                    # print("==============================")
                    # print("selected", thread.get("selected")) #Selected Properties (Comparison Tray) 
                    # print("==============================")

                    # ✅ UPDATE CURRENT THREAD (NO NEW THREAD)
                    thread["selected"] = selected_for_compare.copy() #selected properties dataframe with only rows where Compare column is True (checkbox selected) 
                    thread["comparison_result"] = None
                    thread["comparison_raw"] = None
                    thread["auto_compare_explain"] = False
                    thread["explanation_done"] = False

                    #rename thread
                    selected_ids = selected_for_compare["id"].astype(str).tolist()

                    thread["name"] = "Compare: " + " vs ".join(selected_ids) #gives thread name 

                    # ✅ TRIGGER COMPARISON
                    st.session_state.run_comparison_now = True #trigger when we click the Compare Selected Properties button

                    st.rerun()

        
        # ==============================
        # 🔥 RUN COMPARISON ONLY WHEN TRIGGERED
        # ==============================
        if st.session_state.get("run_comparison_now"):     
            st.session_state.run_comparison_now = False # Once comparison is completed, we turn OFF the trigger
                                                        # to prevent comparison from running again on future Streamlit reruns.

            selected_df = thread.get("selected") #Selected Properties (Comparison Tray) dataframe

            # ✅ ONLY compared rows
            if selected_df is not None and not selected_df.empty and len(selected_df) >= 2:

                compare_df = selected_df.copy() #Create a copy of Selected Properties (Comparison Tray) dataframe

                if "Compare" not in compare_df.columns: 
                    compare_df["Compare"] = True


                render_comparison(
                    recs["input"], #input property dataframe
                    compare_df #selected properties from Selected Properties (Comparison Tray) dataframe
                )

        else:
            # Triggered only when comparison results are already stored in thread memory (generated earlier from comparison_node.py),
            # so comparison graph does not need to run again. 
            if thread.get("comparison_result") is not None: 

                selected_df = thread.get("selected")  #Selected Properties (Comparison Tray) dataframe

                render_comparison(
                    thread.get("comparison_raw"),   
                    selected_df
                )  

        # 💬 Chat ALWAYS after UI
        handle_chat(thread, recs, edited_selected)

# =============================
# RUN APP
# =============================
if __name__ == "__main__":
    main()