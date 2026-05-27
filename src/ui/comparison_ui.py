# ===============================
# comparison_ui.py (FINAL FIXED)
# ===============================

from networkx import display
import streamlit as st
import pydeck as pdk

from src.services.comparison_service import (
    run_comparison,
    prepare_map_data
)

from src.agents.explanation_agent import generate_comparison_explanation
from src.llm.deepseek_client import ask_deepseek_stream

from src.graph.workflow import comparison_graph


# =============================
# CACHE FUNCTION (ONLY ONCE)
# =============================
@st.cache_data(show_spinner=False)
def cached_run_comparison(df, thread_id):
    """
    Cache comparison results to avoid
    recomputing comparison tables.

    Returns:
    - comparison results from run_comparison()
    """
    return run_comparison(df)


# =============================
# MAIN UI FUNCTION
# =============================
def render_comparison(df, edited_selected):
    """
    Render full property comparison UI including:
    - comparison table
    - property map
    - rental estimate
    - AI insights

    Returns:
    - None
    """

    # -----------------------------
    # VALIDATION
    # -----------------------------
    if edited_selected is None:
        st.warning("No properties selected.")
        return

    selected_compare = edited_selected[
        edited_selected["Compare"] == True
    ].copy()  #edited_selected["Compare"] - selected properties dataframe with only rows where Compare column is True (checkbox selected)

    if len(selected_compare) < 2:
        st.warning("Select at least 2 properties to compare")
        return

    # -----------------------------
    # RUN COMPARISON
    # -----------------------------
    active_thread = st.session_state.get("active_thread")

    
    if active_thread:
        thread = st.session_state.threads[active_thread]

        # 🔥 DO NOT recompute if already exists
        # At the first comparison run, thread["comparison_result"] is None from recommendations.py,
        # so the below - "if" block condition becomes False and the "else" block runs always first to generate and save the comparison results.
        if thread.get("comparison_result") is not None:
            raw_df = thread["comparison_raw"] #Selected Properties (Comparison Tray) dataframe (same as compare_df but without the "Compare" column and only the original columns of the properties)
            compare_df = thread["comparison_result"] #Comparison Result dataframe
        #At the FIRST comparison run, this will aways run


        # initial_state is Temporary Runtime State for LangGraph.
        # It exists only while the graph is executing.
        # After graph execution completes, this state gets destroyed,
        # so values stored here are temporary.
        else:
            initial_state = {

                "filters": {},
                "intent": {},
                "slider_weights": {},
                "mode": "dynamic",

                "recommendations": None,

                "selected_properties": selected_compare, #selected properties dataframe with only rows where Compare column is True (checkbox selected)

                "comparison_raw": None,
                "comparison_result": None,

                "explanation": None
            }

            final_state = comparison_graph.invoke(initial_state) #comparison_node and explanation_node get executed inside this comparison_graph
            
            #from comparison_node we get the below two dataframes stored in final_state
            raw_df = final_state["comparison_raw"] 
            compare_df = final_state["comparison_result"]

            explanation = final_state["explanation"] #for this explanation_node get exected in comparison_graph, and the generated explanation is stored in final_state["explanation"]

            # all this threads are already created in thread_manager.py
            thread["comparison_result"] = compare_df
            thread["comparison_raw"] = raw_df
            thread["selected"] = selected_compare.copy() #selected properties dataframe with only rows where Compare column is True (checkbox selected)
                                                        #same dataframme which we have taken above at top as selected_compare and passed to comparison_graph

            thread["last_explanation"] = explanation #this generated using explanation_node in comparison_graph
            thread["show_explanation"] = False # Explanation generated but not shown yet, hence set to False.

    # -----------------------------
    # SAVE TO SESSION
    # -----------------------------
    if active_thread:
        thread = st.session_state.threads[active_thread]
        #this "selected" and "auto_compare_explain" we have created in sidebar.py while creating a new thread
        thread["selected"] = selected_compare.copy() #selected properties dataframe with only rows where Compare column is True (checkbox selected) 

        thread["auto_compare_explain"] = True
        thread["show_explanation"] = True # Allow stored explanation to be displayed.

    # -----------------------------
    # 📊 RESULT TABLE
    # -----------------------------
    st.subheader("📊 Comparison Result")
    st.dataframe(compare_df, width="stretch")

    # ==============================
    # 🗺️ MAP
    # ==============================
    st.subheader("🗺️ Property Map")

    # ✅ FIXED (NO cached_map bug)
    map_df = prepare_map_data(raw_df, df)
    map_df = map_df.dropna(subset=["latitude", "longitude"]).copy()

    if not map_df.empty:

        map_df["price_str"] = map_df["price"].apply(
            lambda x: f"₹{x} Cr"
        )

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position=["longitude", "latitude"],
            get_radius=120,
            get_fill_color=[255, 0, 0, 160],
            pickable=True
        )

        view_state = pdk.ViewState(
            latitude=map_df["latitude"].mean(),
            longitude=map_df["longitude"].mean(),
            zoom=11
        )

        tooltip = {
            "html": "<b>ID:</b> {id} <br/> <b>Price:</b> {price_str}",
            "style": {"backgroundColor": "black", "color": "white"}
        }

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip
            )
        )

    else:
        st.warning("No valid coordinates available")

    # ==============================
    # 💰 RENT TABLE
    # ==============================
    st.subheader("💰 Rental Estimate")

    rent_view = raw_df.loc[:, ["id", "area", "min_rent", "max_rent"]].copy()
    st.dataframe(rent_view, width="stretch")

    # ==============================
    # 🧠 AI INSIGHTS
    # ==============================
    st.subheader("🧠 Property Insights")

    active_thread = st.session_state.get("active_thread")

    if active_thread:
        thread = st.session_state.threads[active_thread]

        # Show stored explanation only when UI display is enabled
        if thread.get("show_explanation"):  
            if thread.get("last_explanation"):
                st.markdown(thread["last_explanation"])




