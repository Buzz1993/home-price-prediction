#price_prediction.py
import json
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import sys
import streamlit as st
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))  # 👈 allows "from scripts..." import

from scripts.target_utils import clean_price_target

data_path = ROOT_DIR / "data" / "raw" / "f_original magicbricks cleaned 12022 data.csv"

# ==============================
# CONFIG
# ==============================
PREDICT_URL = "http://127.0.0.1:8000/predict"
TARGET_COL = "PRICE"


# Columns expected as categorical by model
CATEGORICAL_COLS = [
    "city", "location", "builder", "project_name", "furnish",
    "ownership", "status", "facing", "seller", "flooring",
    "property_type"
]

# Columns expected as numeric
NUMERIC_COLS = [
    "bed", "bath", "balcony", "parking", "lift",
    "area", "available_units", "project_in_acres",
    "flat_on_floor", "total_floor",
    "distance_to_center_km",
    "education_mean_km", "education_min_km",
    "transport_mean_km", "transport_min_km",
    "shopping_centre_mean_km", "shopping_centre_min_km",
    "commercial_hub_mean_km", "commercial_hub_min_km",
    "hospital_mean_km", "hospital_min_km",
    "overall_min_mean_km", "overall_avg_mean_km",
    "overall_min_min_km", "overall_avg_min_km",
    "total_within_2km",
    "bath_bed_ratio", "bed_area_ratio", "bed_bath_ratio",
    "bed_balcony_ratio", "project_density", "compactness_ratio",
    "floor_ratio", "remaining_floors",
    "area_per_bedroom", "area_per_bathroom",
    "area_per_balcony", "area_per_parking",
    "balcony_to_bed_ratio", "parking_to_bed_ratio",
    "lift_to_total_floor_ratio",
    "assigned_amenities_score", "amenities_count"
]


# ==============================
# SANITIZER FUNCTION
# ==============================
def sanitize_input(row_dict):
    clean = {}

    for k, v in row_dict.items():

        # Skip NaN
        if pd.isna(v):
            continue

        # --------------------
        # Force categorical → string
        # --------------------
        if k in CATEGORICAL_COLS:
            clean[k] = str(v).strip().lower()
            continue

        # --------------------
        # Force numeric
        # --------------------
        if k in NUMERIC_COLS:
            try:
                clean[k] = float(v)
            except:
                continue
            continue

        # --------------------
        # Remove lists/dicts (model can't handle)
        # --------------------
        if isinstance(v, (list, dict)):
            continue

        # --------------------
        # Default safe cast
        # --------------------
        if isinstance(v, (np.integer, int)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            clean[k] = float(v)
        else:
            clean[k] = str(v)

    return clean


# ==============================
# MAIN
# ==============================
def main():
    st.title("🏠 Home Price Prediction (Sample Test)")

    df = pd.read_csv(data_path, low_memory=False)
    df = clean_price_target(df, target_col=TARGET_COL)

    if st.button("Run Sample Prediction"):
        sample_row = df.sample(1)
        #print("sample_row",sample_row)
        st.write(sample_row)

        target_val = float(sample_row[TARGET_COL].values.item())

        raw_input = sample_row.drop(columns=[TARGET_COL]).squeeze().to_dict()
        cleaned_input = sanitize_input(raw_input)
        #print("cleaned_input",cleaned_input)
        st.json(cleaned_input)

        st.write(f"📦 Sending **{len(cleaned_input)}** cleaned features to API")

        with st.spinner("Calling prediction API..."):
            response = requests.post(PREDICT_URL, json=cleaned_input, timeout=60)

        if response.status_code == 200:
            result = response.json()

            pred_price = result["predicted_price"]

            col1, col2 = st.columns(2)
            col1.metric("Actual Price (Cr)", round(target_val, 2))
            col2.metric("Predicted Price (Cr)", round(pred_price, 2))

            st.json(result["model_metadata"])

        else:
            st.error("API Error")
            st.json(response.json())




if __name__ == "__main__":
    main()
