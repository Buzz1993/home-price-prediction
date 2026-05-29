# ===============================
# src/graph/state.py
# ===============================

from typing import TypedDict, Optional
import pandas as pd


class PropertyState(TypedDict, total=False):
    # Defines the structure and expected datatypes of the shared LangGraph state.

    # ---------------------------------
    # SYSTEM DATA
    # ---------------------------------
    df: pd.DataFrame
    X_processed: object 

    # ---------------------------------
    # USER INPUT
    # ---------------------------------
    filters: dict
    intent: dict
    slider_weights: dict
    mode: str

    # ---------------------------------
    # PROPERTY SELECTION
    # ---------------------------------
    selected_properties: Optional[pd.DataFrame] #selected_properties can be a a pandas DataFrame or None

    # ---------------------------------
    # SEARCH
    # ---------------------------------
    recommendations: Optional[dict]

    # ---------------------------------
    # COMPARISON
    # ---------------------------------
    comparison_raw: Optional[pd.DataFrame]
    comparison_result: Optional[pd.DataFrame]

    # ---------------------------------
    # EXPLANATION
    # ---------------------------------
    explanation: Optional[str]

    # ---------------------------------
    # ADVISOR
    # ---------------------------------
    advisor_result: object

    # ---------------------------------
    # CHAT SYSTEM
    # ---------------------------------
    user_message: str 
    memory: list 
    route: str
    response: str