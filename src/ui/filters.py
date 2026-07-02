# ===============================
# filters.py
# ===============================

import streamlit as st


# =============================
# DEFAULT FILTER CONFIG
# =============================
def get_default_filters():
    """
    Returns the default values for all property filters.
    """
    return {
        "city": "Any",
        "location": "Any",
        "bed": "Any",
        "furnish": "Any",
        "price_range": "Any",
        "transportation_hubs_clean": "Any",
        "builder": "Any"
    }


# =============================
# INIT FILTER STATE
# =============================
def init_filter_state(default_filters):
    """
    Initializes filter values (usually "Any") in session state if they do not
    already exist.
    """
    for k in default_filters:
        if k not in st.session_state:
            st.session_state[k] = default_filters[k]

    if "last_changed" not in st.session_state:
        st.session_state.last_changed = None


# =============================
# RESET FILTERS
# =============================
def reset_filters(default_filters):
    """
    Resets all filters back to their default values.
    """
    for k in default_filters:
        st.session_state[k] = default_filters[k]

    st.session_state.last_changed = None


# =============================
# APPLY FILTER LOGIC
# =============================
def get_filtered_df(df, filters, exclude_col=None):
    """
    Filters the property data based on the selected filter values.
    """

    temp = df.copy()

    for k, v in filters.items():

        if k == exclude_col:
            continue

        if v != "Any":

            if k in ["builder", "transportation_hubs_clean"]: # For builder and transportation hub filters, partial matching is used so related values are also included.
                temp = temp[temp[k].str.contains(str(v), case=False, na=False)]
            else:
                temp = temp[temp[k] == v]

    return temp


# =============================
# GET DROPDOWN OPTIONS
# =============================
def get_options(df, col, filters):
    """
    Returns available dropdown options based on the current filters.
    """

    temp = get_filtered_df(df, filters, exclude_col=col)

    return ["Any"] + sorted(temp[col].dropna().unique().tolist()) # help to building a dropdown


# =============================
# RENDER FILTER UI
# =============================
def render_filter_ui(df, default_filters):
    # Renders the top filter/input layer of the UI with static/dynamic mode and dropdown filters for- 
    # city, location, bed, furnish, price range, transportation hubs, and builder selection.
    """
    Displays filter dropdowns and handles changes.
    Returns selected filters and mode.
    """

    # MODE
    mode = st.radio("Mode", ["static", "dynamic"])

    # CURRENT FILTERS
    filters = {k: st.session_state[k] for k in default_filters}

    cols = st.columns(len(default_filters))

    for i, k in enumerate(default_filters):

        with cols[i]:

            options = get_options(df, k, filters)
            prev_val = st.session_state[k]

            selected = st.selectbox(
                k,
                options,
                index=options.index(prev_val) if prev_val in options else 0,
                key=k
            )

            # TRACK CHANGE
            if selected != prev_val:
                st.session_state[k] = selected
                st.session_state.last_changed = k

    return filters, mode