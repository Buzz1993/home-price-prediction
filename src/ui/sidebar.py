# ===============================
# sidebar.py
# ===============================

import streamlit as st
import uuid
import pandas as pd
from src.services.thread_manager import create_thread

# =============================
# USER INTENT + SLIDERS
# =============================
def get_user_intent_and_weights(): 
    """
    Collects user preferences from the sidebar and generates
    normalized weights for property ranking.

    Returns:
        tuple: User intent and recommendation weights.

    eg:
    intent = {
        "preferences": ["low budget", "location"]
    }

    weights = {
        "price": 0.30,
        "amenities": 0.10,
        "location": 0.25,
        "area": 0.10,
        "connectivity": 0.15,
        "distance": 0.10
    }
    """

    st.sidebar.markdown("## 🎯 Preferences")

    selected_prefs = st.sidebar.multiselect(
        "Select Preferences",
        ["low budget", "luxury", "location", "spacious", "investment"]
    )

    intent = {"preferences": selected_prefs}

    st.sidebar.markdown("## 🎛️ Fine Tune")

    price_w = st.sidebar.slider("Price", 0.0, 1.0, 0.5)
    amenities_w = st.sidebar.slider("Amenities", 0.0, 1.0, 0.5)
    location_w = st.sidebar.slider("Location", 0.0, 1.0, 0.5)
    area_w = st.sidebar.slider("Area", 0.0, 1.0, 0.5)
    connectivity_w = st.sidebar.slider("Connectivity", 0.0, 1.0, 0.5)
    distance_w = st.sidebar.slider("Distance", 0.0, 1.0, 0.5)

    total = max(price_w + amenities_w + location_w + area_w + connectivity_w + distance_w, 0.0001)

    weights = {
        "price": price_w / total,
        "amenities": amenities_w / total,
        "location": location_w / total,
        "area": area_w / total,
        "connectivity": connectivity_w / total,
        "distance": distance_w / total
    }

    return intent, weights


def render_thread_sidebar():
    # =============================
    # Render thread UI with pin sorting
    # =============================
    """
    Renders the thread sidebar and manages thread actions such as
    creating, opening, renaming, pinning, and deleting chats.
    """

    # if "pinned_threads" not in st.session_state:
    #     st.session_state.pinned_threads = []

    st.sidebar.title("💬 Threads")

    # -----------------------------
    # NEW CHAT
    # -----------------------------
    if st.sidebar.button("➕ New Chat"):
        import uuid
        tid = str(uuid.uuid4())[:8] #generate short unique thread ID example: 'a1b2c3d4' for each new chat thread

        st.session_state.threads[tid] = create_thread() #each thread id and its corresponding data like "messages","data" etc we store in the session_state as session_state is a storage

        # store like 
        # st.session_state
        # |
        # +-- threads
        #     |
        #     +-- a1b2c3d4
        #     |      |
        #     |      +-- messages
        #     |      +-- selected
        #     |      +-- comparison_result
        #     |      +-- explanation
        #     |
        #     +-- x9k2p1m7
        #         |
        #         +-- messages
        #         +-- selected

        st.session_state.active_thread = tid                    #This means - Current opened thread = newly created thread

    st.sidebar.markdown("---")

    # -----------------------------
    # SORT (PIN ON TOP)
    # -----------------------------
    pinned = []
    unpinned = []

    for tid, tdata in st.session_state.threads.items():
        if tid in st.session_state.pinned_threads:
            pinned.append((tid, tdata))
        else:
            unpinned.append((tid, tdata))

    sorted_threads = pinned + unpinned

    # -----------------------------
    # RENDER THREADS
    # -----------------------------
    for tid, tdata in sorted_threads:

        """
        Renders all chat threads in the sidebar.

        Features:
        - Displays pinned threads at the top
        - Opens selected thread
        - Allows thread rename
        - Supports pin/unpin functionality
        - Limits pinned threads to 5
        - Deletes threads safely
        - Updates active thread state
        - Reruns Streamlit UI after actions
        """

        col1, col2 = st.sidebar.columns([6, 1])

        label = f"📌 {tdata['name']}" if tid in st.session_state.pinned_threads else tdata["name"]

        # OPEN THREAD
        if col1.button(label, key=f"open_{tid}"):
            st.session_state.active_thread = tid

        # ACTIONS
        with col2:
            with st.popover("⋯"):

                new_name = st.text_input(
                    "Rename",
                    value=tdata["name"],
                    key=f"rename_input_{tid}"
                )

                if st.button("✏️ Rename", key=f"rename_btn_{tid}"):
                    st.session_state.threads[tid]["name"] = new_name
                    print("*********rename got clicked***********") #when click on the rename then only this get print
                    st.rerun()

                # PIN / UNPIN
                if tid in st.session_state.pinned_threads:
                    if st.button("📌 Unpin", key=f"unpin_{tid}"):
                        st.session_state.pinned_threads.remove(tid)
                        print("*********Unpin clicked***********")
                        st.rerun()
                else:
                    if st.button("📌 Pin", key=f"pin_{tid}"):
                        if len(st.session_state.pinned_threads) >= 5:
                            st.warning("Max 5 pinned threads allowed")
                        else:
                            st.session_state.pinned_threads.append(tid)
                        print("*********Pin clicked***********")    
                        st.rerun()

                # DELETE
                if st.button("🗑️ Delete", key=f"delete_{tid}"):
                    st.session_state.threads.pop(tid, None)

                    if tid in st.session_state.pinned_threads:
                        st.session_state.pinned_threads.remove(tid)

                    if st.session_state.active_thread == tid:
                        st.session_state.active_thread = None

                    print("*********Delete clicked***********")                   
                    st.rerun() #rerun entire Streamlit script using latest session_state values to rebuild/update UI
                               #because of rerun main() executes again from top and this main() is the function from recommendations.py