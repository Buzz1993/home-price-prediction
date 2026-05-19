# # ===============================
# # prediction_node.py
# # ===============================

# from src.services.prediction_service import (
#     predict_property_price
# )

# from src.llm.deepseek_client import ask_deepseek


# def prediction_node(state):

#     print("✅ prediction_node executed")

#     selected_df = state.get("selected_properties")

#     # ---------------------------------
#     # VALIDATION
#     # ---------------------------------
#     if selected_df is None or selected_df.empty:

#         state["response"] = (
#             "No selected property available for prediction."
#         )

#         return state

#     # ---------------------------------
#     # USE FIRST PROPERTY
#     # ---------------------------------
#     property_row = selected_df.iloc[0]

#     prediction_result = predict_property_price(property_row)

#     # ---------------------------------
#     # API FAILURE
#     # ---------------------------------
#     if not prediction_result["success"]:

#         state["response"] = (
#             f"Prediction failed: "
#             f"{prediction_result['error']}"
#         )

#         return state

#     # ---------------------------------
#     # EXTRACT PREDICTION
#     # ---------------------------------
#     prediction_data = prediction_result["prediction"]

#     predicted_price = prediction_data.get(
#         "predicted_price",
#         "Unknown"
#     )

#     property_id = property_row.get("id", "Unknown")

#     locality = property_row.get("location", "Unknown")

#     bhk = property_row.get("bed", "Unknown")

#     # ---------------------------------
#     # LLM EXPLANATION
#     # ---------------------------------
#     prompt = f"""
#     You are an expert Indian real estate analyst.

#     PROPERTY:
#     - ID: {property_id}
#     - Location: {locality}
#     - BHK: {bhk}

#     ML MODEL PREDICTION:
#     Predicted Price = ₹{predicted_price} Cr

#     TASK:
#     Explain the prediction in simple practical language.

#     IMPORTANT:
#     - Use short explanation
#     - Do NOT hallucinate
#     - Do NOT invent amenities
#     - Do NOT invent infrastructure
#     - Prices are in INR
#     """

#     explanation = ask_deepseek(prompt)

#     # ---------------------------------
#     # FINAL RESPONSE
#     # ---------------------------------
#     state["response"] = f"""
# 🏠 PROPERTY PRICE PREDICTION

# Property ID: {property_id}

# Predicted Price: ₹{predicted_price} Cr

# Explanation:
# {explanation}
# """

#     return state


#==============================================================================================================================================================================


# # ===============================
# # prediction_node.py
# # ===============================

# from src.services.prediction_service import (
#     predict_property_price
# )

# from src.llm.deepseek_client import ask_deepseek


# def prediction_node(state):

#     print("✅ prediction_node executed")

#     selected_df = state.get("selected_properties")

#     # ---------------------------------
#     # VALIDATION
#     # ---------------------------------
#     if selected_df is None or selected_df.empty:

#         state["response"] = (
#             "No selected property available for prediction."
#         )

#         return state

#     # ---------------------------------
#     # FIND REQUESTED PROPERTY
#     # ---------------------------------

#     user_msg = state.get("user_message", "").lower()

#     property_row = None

#     # ---------------------------------
#     # CASE 1: User mentioned property ID
#     # ---------------------------------
#     for _, row in selected_df.iterrows():

#         row_id = str(row.get("id", "")).lower()

#         if row_id and row_id in user_msg:
#             property_row = row
#             break

#     # ---------------------------------
#     # CASE 2: User asked for WINNER
#     # ---------------------------------
#     if property_row is None:

#         if "winner" in user_msg or "best property" in user_msg:

#             comparison_df = state.get("comparison_result")

#             if comparison_df is not None and not comparison_df.empty:

#                 best_row = comparison_df.sort_values(
#                     "overall_score",
#                     ascending=False
#                 ).iloc[0]

#                 best_id = str(best_row["id"]).lower()

#                 matched = selected_df[
#                     selected_df["id"].astype(str).str.lower() == best_id
#                 ]

#                 if not matched.empty:
#                     property_row = matched.iloc[0]

#     # ---------------------------------
#     # CASE 3: Predict ALL properties
#     # ---------------------------------
#     if property_row is None and (
#         "all properties" in user_msg
#         or "selected properties" in user_msg
#         or "all selected" in user_msg
#     ):

#         responses = []

#         for _, row in selected_df.iterrows():

#             prediction_result = predict_property_price(row)

#             if prediction_result["success"]:

#                 pred_price = prediction_result["prediction"].get(
#                     "predicted_price",
#                     "Unknown"
#                 )

#                 responses.append(
#                     f"• {row['id']} → ₹{pred_price} Cr"
#                 )

#         state["response"] = (
#             "🏠 PROPERTY PRICE PREDICTIONS\n\n"
#             + "\n".join(responses)
#         )

#         return state

#     # ---------------------------------
#     # DEFAULT FALLBACK
#     # ---------------------------------
#     if property_row is None:
#         property_row = selected_df.iloc[0]

#     print("\n========== PROPERTY ROW COLUMNS ==========")
#     print(property_row.index.tolist())
#     print("=========================================\n")

#     # ---------------------------------
#     # HANDLE BOTH id / ID
#     # ---------------------------------
#     property_id = None

#     if "id" in property_row.index:
#         property_id = property_row["id"]

#     elif "ID" in property_row.index:
#         property_id = property_row["ID"]

#     else:
#         state["response"] = (
#             "Prediction failed: property ID column not found."
#         )
#         return state

#     print("PROPERTY ID:", property_id)

#     prediction_result = predict_property_price(property_row)

#     # ---------------------------------
#     # API FAILURE
#     # ---------------------------------
#     if not prediction_result["success"]:

#         state["response"] = (
#             f"Prediction failed: "
#             f"{prediction_result['error']}"
#         )

#         return state

#     # ---------------------------------
#     # EXTRACT PREDICTION
#     # ---------------------------------
#     prediction_data = prediction_result["prediction"]

#     predicted_price = prediction_data.get(
#         "predicted_price",
#         "Unknown"
#     )

#     property_id = (
#         property_row.get("id")
#         if "id" in property_row.index
#         else property_row.get("ID", "Unknown")
#     )

#     locality = property_row.get("location", "Unknown")

#     bhk = property_row.get("bed", "Unknown")

#     # ---------------------------------
#     # LLM EXPLANATION
#     # ---------------------------------
#     prompt = f"""
#     You are a deterministic Indian real estate analyst.

#     STRICT RULES:
#     - Use ONLY provided property data
#     - NEVER hallucinate
#     - NEVER invent amenities
#     - NEVER invent infrastructure
#     - NEVER change conclusions randomly
#     - Keep answer short and consistent
#     - Same input must produce same explanation
#     - Prices are in INR only

#     PROPERTY:
#     - ID: {property_id}
#     - Location: {locality}
#     - BHK: {bhk}

#     MODEL PREDICTION:
#     ₹{predicted_price} Cr

#     Explain ONLY:
#     1. Location impact
#     2. Property size impact
#     3. Market positioning

#     Maximum 5 bullet points.
#     """

#     explanation = ask_deepseek(prompt)

#     # ---------------------------------
#     # FINAL RESPONSE
#     # ---------------------------------
#     state["response"] = f"""
# 🏠 PROPERTY PRICE PREDICTION

# Property ID: {property_id}

# Predicted Price: ₹{predicted_price} Cr

# Explanation:
# {explanation}
# """

#     return state

#===============================================================================================================================================================================

# ===============================
# prediction_node.py (REFACTORED + DOCSTRINGS)
# ===============================

from src.services.prediction_service import predict_property_price
from src.llm.deepseek_client import ask_deepseek


# =========================================
# VALIDATE SELECTED DATA
# =========================================
def validate_selected_properties(selected_df):
    """
    Validate whether selected properties exist.

    Parameters:
        selected_df (pd.DataFrame):
            DataFrame containing selected properties.

    Returns:
        tuple:
            (True, None) if valid,
            otherwise (False, error_message)
    """

    if selected_df is None or selected_df.empty:
        return False, "No selected property available for prediction."

    return True, None


# =========================================
# FIND PROPERTY USING PROPERTY ID
# =========================================
def find_property_by_id(selected_df, user_msg):
    """
    Find property row if user mentioned
    property ID in message.

    Parameters:
        selected_df (pd.DataFrame):
            Selected properties dataframe.

        user_msg (str):
            User message in lowercase.

    Returns:
        pd.Series or None:
            Matching property row if found.
    """

    for _, row in selected_df.iterrows():

        row_id = str(
            row.get("id", "")
        ).strip().lower()

        if row_id and row_id in user_msg:
            return row

    return None


# =========================================
# FIND BEST/WINNER PROPERTY
# =========================================
def find_best_property(
    selected_df,
    comparison_df
):
    """
    Find best property based on
    highest overall_score from
    comparison dataframe.

    Parameters:
        selected_df (pd.DataFrame):
            Selected properties dataframe.

        comparison_df (pd.DataFrame):
            Comparison result dataframe.

    Returns:
        pd.Series or None:
            Best matching property row.
    """

    if comparison_df is None or comparison_df.empty:
        return None

    best_row = comparison_df.sort_values(
        "overall_score",
        ascending=False
    ).iloc[0]

    best_id = str(
        best_row["id"]
    ).strip().lower()

    matched = selected_df[
        selected_df["id"]
        .astype(str)
        .str.strip()
        .str.lower()
        == best_id
    ]

    if matched.empty:
        return None

    return matched.iloc[0]


# =========================================
# CHECK IF USER ASKED ALL PROPERTIES
# =========================================
def is_all_prediction_request(user_msg):
    """
    Check whether user wants prediction
    for all selected properties.

    Parameters:
        user_msg (str):
            User message in lowercase.

    Returns:
        bool:
            True if all-properties prediction requested.
    """

    keywords = [
        "all properties",
        "selected properties",
        "all selected"
    ]

    return any(k in user_msg for k in keywords)


# =========================================
# PREDICT ALL PROPERTIES
# =========================================
def predict_all_properties(selected_df):
    """
    Predict prices for all selected properties.

    Parameters:
        selected_df (pd.DataFrame):
            Selected properties dataframe.

    Returns:
        str:
            Formatted prediction response text.
    """

    responses = []

    for _, row in selected_df.iterrows():

        prediction_result = predict_property_price(row)

        # ---------------------------------
        # FAILED PREDICTION
        # ---------------------------------
        if not prediction_result["success"]:

            responses.append(
                f"""
❌ Property ID: {row.get('id', 'Unknown')}

Prediction Failed:
{prediction_result['error']}
"""
            )

            continue

        # ---------------------------------
        # EXTRACT VALUES
        # ---------------------------------
        (
            property_id,
            original_price,
            predicted_price
        ) = extract_prediction_details(
            row,
            prediction_result
        )

        # ---------------------------------
        # APPEND RESPONSE
        # ---------------------------------
        responses.append(
            f"""
🏠 Property ID: {property_id}

Original Price: ₹{original_price} Cr

Predicted Price: ₹{predicted_price} Cr
"""
        )

    return (
        "🏠 ALL PROPERTY PREDICTIONS\n\n"
        + "\n---------------------------\n".join(responses)
    )


# =========================================
# GET PROPERTY ID SAFELY
# =========================================
def get_property_id(property_row):
    """
    Safely extract property ID from row.

    Supports both:
    - id
    - ID

    Parameters:
        property_row (pd.Series):
            Property row.

    Returns:
        str or None:
            Property ID if found.
    """

    if "id" in property_row.index:
        return property_row["id"]

    if "ID" in property_row.index:
        return property_row["ID"]

    return None


# =========================================
# GENERATE EXPLANATION
# =========================================
def generate_prediction_explanation(
    property_id,
    locality,
    bhk,
    predicted_price
):
    """
    Generate deterministic LLM explanation
    for predicted property price.

    Parameters:
        property_id (str):
            Property ID.

        locality (str):
            Property location.

        bhk (str/int):
            BHK count.

        predicted_price (float/str):
            Predicted property price.

    Returns:
        str:
            LLM-generated explanation.
    """

    prompt = f"""
    You are a deterministic Indian real estate analyst.

    STRICT RULES:
    - Use ONLY provided property data
    - NEVER hallucinate
    - NEVER invent amenities
    - NEVER invent infrastructure
    - NEVER change conclusions randomly
    - Keep answer short and consistent
    - Same input must produce same explanation
    - Prices are in INR only

    PROPERTY:
    - ID: {property_id}
    - Location: {locality}
    - BHK: {bhk}

    MODEL PREDICTION:
    ₹{predicted_price} Cr

    Explain ONLY:
    1. Location impact
    2. Property size impact
    3. Market positioning

    Maximum 5 bullet points.
    """

    return ask_deepseek(prompt)


# =========================================
# BUILD FINAL RESPONSE
# =========================================
def build_prediction_response(
    property_id,
    original_price,
    predicted_price,
    explanation
):
    """
    Build final formatted prediction response.

    Parameters:
        property_id (str):
            Property ID.

        original_price (float/str):
            Original property price.

        predicted_price (float/str):
            Predicted property price.

        explanation (str):
            LLM explanation.

    Returns:
        str:
            Final formatted response text.
    """

    return f"""
🏠 PROPERTY PRICE PREDICTION

Property ID: {property_id}

Original Price: ₹{original_price} Cr

Predicted Price: ₹{predicted_price} Cr

Price Difference: ₹{round(float(predicted_price) - float(original_price), 2)} Cr

Explanation:
{explanation}
"""

# =========================================
# EXTRACT PREDICTION DETAILS
# =========================================
def extract_prediction_details(
    property_row,
    prediction_result
):
    """
    Extract commonly used prediction details.

    Parameters:
        property_row (pd.Series):
            Property dataframe row.

        prediction_result (dict):
            Prediction API response.

    Returns:
        tuple:
            property_id,
            original_price,
            predicted_price
    """

    prediction_data = prediction_result[
        "prediction"
    ]

    predicted_price = prediction_data.get(
        "predicted_price",
        "Unknown"
    )

    original_price = property_row.get(
        "price",
        property_row.get("PRICE", "Unknown")
    )

    property_id = property_row.get(
        "id",
        property_row.get("ID", "Unknown")
    )

    return (
        property_id,
        original_price,
        predicted_price
    )


# =========================================
# MAIN NODE
# =========================================
def prediction_node(state):
    """
    Main prediction node for LangGraph.

    Responsibilities:
    - Validate selected properties
    - Detect requested property
    - Handle best-property prediction
    - Handle all-property prediction
    - Run prediction API
    - Generate explanation
    - Store final response in state

    Parameters:
        state (dict):
            LangGraph state dictionary.

    Returns:
        dict:
            Updated LangGraph state.
    """

    print("✅ prediction_node executed")

    selected_df = state.get(
        "selected_properties"
    )

    # =========================================
    # VALIDATION
    # =========================================
    is_valid, error = validate_selected_properties(
        selected_df
    )

    if not is_valid:

        state["response"] = error

        return state

    user_msg = state.get(
        "user_message",
        ""
    ).lower()

    property_row = None

    # =========================================
    # CASE 1: PROPERTY ID
    # =========================================
    property_row = find_property_by_id(
        selected_df,
        user_msg
    )

    # =========================================
    # CASE 2: WINNER / BEST PROPERTY
    # =========================================
    if property_row is None:

        if (
            "winner" in user_msg
            or "best property" in user_msg
        ):

            property_row = find_best_property(
                selected_df,
                state.get("comparison_result")
            )

    # =========================================
    # CASE 3: ALL PROPERTIES
    # =========================================
    if (
        property_row is None
        and is_all_prediction_request(user_msg)
    ):

        state["response"] = predict_all_properties(
            selected_df
        )

        return state

    # =========================================
    # DEFAULT FALLBACK
    # =========================================
    if property_row is None:

        property_row = selected_df.iloc[0]

    print("\n========== PROPERTY ROW ==========")
    print(property_row)
    print("=================================\n")

    # =========================================
    # GET PROPERTY ID
    # =========================================
    property_id = get_property_id(
        property_row
    )

    if property_id is None:

        state["response"] = (
            "Prediction failed: "
            "property ID column not found."
        )

        return state

    print("PROPERTY ID:", property_id)

    # =========================================
    # RUN PREDICTION
    # =========================================
    prediction_result = predict_property_price(
        property_row
    )

    # =========================================
    # API FAILURE
    # =========================================
    if not prediction_result["success"]:

        state["response"] = (
            f"Prediction failed: "
            f"{prediction_result['error']}"
        )

        return state

    # =========================================
    # EXTRACT PREDICTION DETAILS
    # =========================================
    (
        property_id,
        original_price,
        predicted_price
    ) = extract_prediction_details(
        property_row,
        prediction_result
    )

    locality = property_row.get(
        "location",
        "Unknown"
    )

    bhk = property_row.get(
        "bed",
        "Unknown"
    )

    # =========================================
    # GENERATE LLM EXPLANATION
    # =========================================
    explanation = generate_prediction_explanation(
        property_id=property_id,
        locality=locality,
        bhk=bhk,
        predicted_price=predicted_price
    )

    # =========================================
    # FINAL RESPONSE
    # =========================================
    state["response"] = build_prediction_response(
        property_id=property_id,
        original_price=original_price,
        predicted_price=predicted_price,
        explanation=explanation
    )

    return state