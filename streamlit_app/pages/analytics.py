#analytics.py
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# -------------------------------------------------
# Fix path so Streamlit can access project root
# -------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from scripts.target_utils import clean_price_target

DATA_PATH = ROOT_DIR / "data" / "raw" / "f_original magicbricks cleaned 12022 data.csv"
TARGET_COL = "PRICE"

st.set_page_config(layout="wide")
st.title("📊 Real Estate Market Analytics")
st.markdown("Explore property price trends and patterns from the dataset.")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = clean_price_target(df, target_col=TARGET_COL)
    return df

df = load_data()

# -------------------------------------------------
# Detect useful columns dynamically
# -------------------------------------------------
def find_column(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

city_col = find_column(["city", "City", "addressRegion", "city_name"])
bed_col = find_column(["bed", "beds", "bedrooms"])
area_col = find_column(["area", "super_built_up_area", "carpet_area"])

# ---------------------------
# Sidebar Filters
# ---------------------------
st.sidebar.header("🔍 Filters")

if city_col:
    cities = sorted(df[city_col].dropna().unique().tolist())
    selected_city = st.sidebar.selectbox("Select City", ["All"] + cities)

    if selected_city != "All":
        df = df[df[city_col] == selected_city]
else:
    st.sidebar.info("City filter not available in dataset")

# ---------------------------
# Summary Metrics
# ---------------------------
st.subheader("📌 Market Summary")

colA, colB, colC = st.columns(3)
colA.metric("Avg Price (Cr)", round(df[TARGET_COL].mean(), 2))
colB.metric("Max Price (Cr)", round(df[TARGET_COL].max(), 2))
colC.metric("Min Price (Cr)", round(df[TARGET_COL].min(), 2))

# ---------------------------
# Avg Price by Bedrooms
# ---------------------------
st.subheader("🏠 Avg Price by Bedrooms")

if bed_col:
    avg_bhk = df.groupby(bed_col)[TARGET_COL].mean().sort_index()
    st.bar_chart(avg_bhk)
else:
    st.info("Bedroom data not available")

# ---------------------------
# Price Distribution
# ---------------------------
st.subheader("💰 Price Distribution")

price_counts = df[TARGET_COL].round(1).value_counts().sort_index()
st.bar_chart(price_counts)

# ---------------------------
# Area vs Price
# ---------------------------
st.subheader("📐 Area vs Price")

if area_col:
    scatter_df = df[[area_col, TARGET_COL]].dropna()
    st.scatter_chart(scatter_df, x=area_col, y=TARGET_COL)
else:
    st.info("Area data not available")

# ---------------------------
# Avg Price by City
# ---------------------------
st.subheader("🌆 Average Price by City")

if city_col:
    city_price = df.groupby(city_col)[TARGET_COL].mean().sort_values(ascending=False)
    st.bar_chart(city_price)
else:
    st.info("City-wise analysis not available")
