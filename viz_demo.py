import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scripts.data_clean_utils import perform_property_data_cleaning

st.set_page_config(page_title="Mumbai Home Price Predictor", layout="wide")

# =====================================================
# PATHS & MODEL LOADING
# =====================================================
ARTIFACTS_DIR = Path("cache/artifacts_auto")

@st.cache_resource
def load_ml_components():
    # Load Swiggy-style artifacts
    cleaner = joblib.load(ARTIFACTS_DIR / "cleaner_full_for_te.pkl")
    te_full = joblib.load(ARTIFACTS_DIR / "te_full.pkl")
    preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
    model = joblib.load(ARTIFACTS_DIR / "final_model.joblib")
    return cleaner, te_full, preprocessor, model

try:
    cleaner, te_full, preprocessor, model = load_ml_components()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# =====================================================
# UI LAYOUT
# =====================================================
st.title("🏙️ Mumbai Home Price Prediction")

with st.sidebar:
    st.header("Location Details")
    project_name = st.text_input("Project Name", "Lodha Palava")
    builder = st.text_input("Builder Name", "Lodha Group")
    address_locality = st.text_input("Locality", "Kharghar")
    city_input = st.selectbox("City", ["mumbai", "navi mumbai", "thane", "palghar"])

col1, col2, col3 = st.columns(3)
with col1:
    bhk = st.number_input("BHK (Bedrooms)", min_value=1.0, value=2.0)
    bathrooms = st.number_input("Bathrooms", min_value=1.0, value=2.0)
    area_val = st.number_input("Area (Sq.Ft)", min_value=100.0, value=1000.0)

with col2:
    status = st.selectbox("Current Status", ["ready to move", "under construction"])
    furnish = st.selectbox("Furnishing", ["unfurnished", "semi-furnished", "furnished"])
    prop_type = st.selectbox("Transaction Type", ["new property", "resale"])

with col3:
    total_floors = st.number_input("Total Floors", value=20.0)
    flat_floor = st.number_input("Property Floor", value=5.0)
    parking = st.number_input("Parking Spots", value=1.0)

# =====================================================
# PREDICTION LOGIC
# =====================================================
if st.button("Predict Price", type="primary"):
    # CONSTRUCT RAW INPUT: Satisfies every KeyError and AttributeError possible
    raw_input = {
        "name": f"{bhk} BHK Flat in {project_name}",
        "geo": "{'latitude': 19.0, 'longitude': 72.0}", 
        "address": str({"addresslocality": address_locality, "addressregion": city_input}),
        "md_address": f"{address_locality}, {city_input}",
        "area": f"{area_val} sqft",
        "many_carpet area": f"{area_val} sqft",
        "many_floor": f"{flat_floor} ({total_floors})",
        
        # Bed/Bath Consolidation Placeholders
        "numberofrooms": bhk,
        "bb_beds": np.nan, "leftbb_beds": np.nan, "bb_bed": np.nan, "leftbb_bed": np.nan,
        "bb_baths": bathrooms,
        "leftbb_baths": np.nan, "bb_bath": np.nan, "leftbb_bath": np.nan,
        
        # Lift Consolidation Placeholders (Fixes Line 360 KeyError)
        "md_lift": 1.0,
        "many_lifts": np.nan, "leftmany_lifts": np.nan, "many_lift": np.nan, "leftmany_lift": np.nan,

        # Category Placeholders
        "many_developer": builder,
        "ap_pjt_name": project_name,
        "many_transaction type": prop_type,
        "many_status": status,
        "many_furnished status": furnish,
        "bb_covered-parking": parking,
        "md_age of construction": "new construction",
        "potentialaction": "dummy:dummy:Owner",
        "md_additional rooms": "none of these",
        "md_floors allowed for construction": total_floors,
        "leftmany_floor": "",

        # Catch-all for remaining columns accessed in basic_cleaning
        **{col: "" for col in ["leftmany_developer", "ap_buildr", "many_project", 
                               "leftmany_project", "leftmany_transaction type", 
                               "leftmany_status", "many_age of construction", 
                               "leftmany_age of construction", "leftmany_furnished status", 
                               "leftmany_facing", "many_facing", "leftmany_additional rooms",
                               "bb_balcony", "leftbb_balcony", "bb_balconies", "leftbb_balconies",
                               "leftmany_carpet area", "many_super built-up area", 
                               "leftmany_super built-up area", "md_overlooking", "md_flooring",
                               "locality_url", "@id", "@type", "bhk_type", "property_loc",
                               "md_water availability", "md_status of electricity", "md_landmarks",
                               "md_authority approval", "md_rera id", "aboutpjt_launch date", 
                               "aboutpjt_project size", "aboutpjt_total units", "aboutpjt_total towers"]}
    }
    
    # Initialize AM_ columns
    for i in range(12000, 15000): raw_input[f"am_{i}"] = ""

    X_raw = pd.DataFrame([raw_input])
    
    # Run modular cleaning
    X_clean = perform_property_data_cleaning(X_raw)
    
    # Apply Target Encoding artifacts
    te_cols = ["builder", "project_name", "location"]
    for col in te_cols:
        if col not in X_clean.columns: X_clean[col] = "missing"
    
    X_te_in = cleaner.transform(X_clean[te_cols])
    X_te_out = te_full.transform(X_te_in)
    for col in te_cols: X_clean[f"{col}_te"] = X_te_out[col].values
    X_clean.drop(columns=te_cols, inplace=True)

    # Preprocess and Predict
    X_prepped = preprocessor.transform(X_clean)
    final_price = np.expm1(model.predict(X_prepped)[0])

    st.balloons()
    st.success(f"### Estimated Price: ₹ {round(final_price, 2)} Cr")