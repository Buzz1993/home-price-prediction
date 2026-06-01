# ===============================
# prediction_tool.py
# ===============================

from src.services.prediction_service import (
    predict_property_price
)

import pandas as pd


def predict_price(
    property_id: str
):

    df = pd.read_csv(
        "data/raw/f_original magicbricks cleaned 12022 data.csv"
    )

    row = df[
        df["id"] == property_id
    ].iloc[0]

    result = predict_property_price(
        row
    )

    return result