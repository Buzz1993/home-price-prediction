#main.py
# import streamlit as st

# # Page config
# st.set_page_config(
#     page_title="Home Price Prediction System",
#     page_icon="🏠",
#     layout="wide"
# )

# # Title
# st.title("🏠 Home Price Prediction & Insights Platform")

# st.markdown("""
# Welcome to the **End-to-End ML Powered Real Estate Platform**.

# This application is built using a **production-style architecture**:

# - ⚙️ **FastAPI Backend** → Serves the trained ML model  
# - 🧠 **Machine Learning Pipeline** → Feature engineering + preprocessing + model  
# - 🎨 **Streamlit Frontend** → User interface for predictions and insights  

# Use the navigation panel on the left to explore different features of the system.
# """)

# st.divider()

# # Sections
# col1, col2, col3 = st.columns(3)

# with col1:
#     st.subheader("💰 Price Prediction")
#     st.write("""
#     Predict the market price of a property using our trained machine learning model.
    
#     ✔ Real-time API prediction  
#     ✔ Production deployed model  
#     ✔ Advanced preprocessing pipeline
#     """)

# with col2:
#     st.subheader("📊 Market Analytics")
#     st.write("""
#     Explore housing market trends and insights.
    
#     ✔ Price distributions  
#     ✔ Location-based analysis  
#     ✔ Feature impact visualization
#     """)

# with col3:
#     st.subheader("🏘 Property Recommendations")
#     st.write("""
#     Get smart suggestions based on property features and predicted price.
    
#     ✔ Similar property ideas  
#     ✔ Budget-based suggestions  
#     ✔ Location-aware insights
#     """)

# st.divider()

# # System Info Section
# st.subheader("🛠️ System Architecture")

# st.markdown("""
# This project follows a **real-world ML system design**:


# - The **model is NOT inside Streamlit**
# - Streamlit communicates with the backend using API calls
# - The backend loads model artifacts from MLflow & DVC

# This makes the system:
# - Scalable
# - Modular
# - Production-ready
# """)

# st.divider()

# st.success("👈 Use the sidebar to start with Price Prediction, Analytics, or Recommendations!")

#main.py
import streamlit as st

st.set_page_config(page_title="Home Price Prediction", page_icon="🏠", layout="wide")

st.title("🏠 Home Price Prediction System")
st.markdown("""
Welcome to the **AI Powered Property Price Predictor**

Use the sidebar to navigate to the **Price Prediction** page and get an instant estimate.
""")

st.info("Model predicts property price in Crores (₹ Cr)")

