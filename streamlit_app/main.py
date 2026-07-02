# ===============================
# main.py
# ===============================

import sys

# Configure UTF-8 output for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
    
import streamlit as st

st.set_page_config(page_title="Home Price Prediction", page_icon="🏠", layout="wide")

st.title("🏠 Home Price Prediction System")
st.markdown("""
Welcome to the **AI Powered Property Price Predictor**

Use the sidebar to navigate to the **Price Prediction** page and get an instant estimate.
""")

st.info("Model predicts property price in Crores (₹ Cr)")

