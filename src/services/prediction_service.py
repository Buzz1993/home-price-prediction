# # ===============================
# # prediction_service.py
# # ===============================

# import requests
# import pandas as pd


# PREDICT_API_URL = "http://127.0.0.1:8000/predict"


# # ===============================
# # SAFE PAYLOAD CLEANER
# # ===============================
# def clean_payload(row_dict):

#     clean = {}

#     # -----------------------------
#     # SAME LOGIC AS WORKING PAGE
#     # -----------------------------
#     categorical_cols = [
#         "city", "location", "builder",
#         "project_name", "furnish",
#         "ownership", "status",
#         "facing", "seller",
#         "flooring", "property_type"
#     ]

#     numeric_cols = [
#         "bed", "bath", "balcony",
#         "parking", "lift", "area"
#     ]

#     for k, v in row_dict.items():

#         # -----------------------------
#         # SKIP NULLS
#         # -----------------------------
#         if pd.isna(v):
#             continue

#         # -----------------------------
#         # REMOVE LISTS / DICTS
#         # VERY IMPORTANT
#         # -----------------------------
#         if isinstance(v, (list, dict)):
#             continue

#         # -----------------------------
#         # CATEGORICAL
#         # -----------------------------
#         if k in categorical_cols:
#             clean[k] = str(v).strip().lower()
#             continue

#         # -----------------------------
#         # NUMERIC
#         # -----------------------------
#         if k in numeric_cols:
#             try:
#                 clean[k] = float(v)
#             except:
#                 continue
#             continue

#         # -----------------------------
#         # SAFE NUMERIC CAST
#         # -----------------------------
#         if isinstance(v, (int, float)):
#             clean[k] = v

#         else:
#             clean[k] = str(v)

#     return clean


# def predict_property_price(property_row):

#     try:

#         payload = clean_payload(
#             property_row.to_dict()
#         )

#         print("\n========== PAYLOAD ==========")
#         print(payload)
#         print("=============================\n")

#         response = requests.post(
#             PREDICT_API_URL,
#             json=payload,
#             timeout=60
#         )

#         print("STATUS:", response.status_code)
#         print("RESPONSE:", response.text)

#         if response.status_code != 200:

#             return {
#                 "success": False,
#                 "error": f"Prediction API failed ({response.status_code})"
#             }

#         data = response.json()

#         return {
#             "success": True,
#             "prediction": data
#         }

#     except Exception as e:

#         return {
#             "success": False,
#             "error": str(e)
#         }

#=============================================================================================================================================================================

# ===============================
# prediction_service.py
# ===============================

import requests
import pandas as pd
from pathlib import Path
import sys
import traceback
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))


PREDICT_API_URL = "http://127.0.0.1:8000/predict"

RAW_DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "f_original magicbricks cleaned 12022 data.csv"
)

# ==========================================
# SAFE JSON SANITIZER
# ==========================================
def sanitize_payload(row_dict):

    clean = {}

    for k, v in row_dict.items():

        # ---------------------------------
        # REMOVE NaN / None
        # ---------------------------------
        if pd.isna(v):
            continue

        # ---------------------------------
        # REMOVE inf / -inf
        # ---------------------------------
        if isinstance(v, (float, np.floating)):

            if np.isinf(v):
                continue

        # ---------------------------------
        # REMOVE list/dict
        # ---------------------------------
        if isinstance(v, (list, dict)):
            continue

        # ---------------------------------
        # SAFE NUMERIC CAST
        # ---------------------------------
        if isinstance(v, (np.integer, int)):
            clean[k] = int(v)

        elif isinstance(v, (np.floating, float)):
            clean[k] = float(v)

        else:
            clean[k] = str(v)

    return clean



def predict_property_price(property_row):

    try:

        # print("\n========== PROPERTY ROW ==========")
        # print(property_row)
        # print("=================================\n")

        # print("\n========== PROPERTY ROW INDEX ==========")
        # print(property_row.index.tolist())
        # print("=======================================\n")

        # ==========================================
        # LOAD ORIGINAL RAW DATA
        # ==========================================

        raw_df = pd.read_csv(
            RAW_DATA_PATH,
            low_memory=False
        )

        # print("\n========== RAW DF COLUMNS ==========")
        # print(raw_df.columns.tolist())
        # print("===================================\n")

        if "id" in property_row.index:
            property_id = property_row["id"]

        elif "ID" in property_row.index:
            property_id = property_row["ID"]

        else:
            return {
                "success": False,
                "error": "No property ID column found"
            }

        property_id = str(property_id).strip().lower()


        # ==========================================
        # HANDLE RAW CSV ID COLUMN SAFELY
        # ==========================================
        raw_df.columns = raw_df.columns.str.strip()

        # print("\n========== RAW DF CLEANED COLUMNS ==========")
        # print(raw_df.columns.tolist())
        # print("===========================================\n")

        raw_id_col = None

        if "id" in raw_df.columns:
            raw_id_col = "id"

        elif "ID" in raw_df.columns:
            raw_id_col = "ID"

        else:
            return {
                "success": False,
                "error": f"No ID column found in raw CSV. Columns: {raw_df.columns.tolist()}"
            }

        # print("RAW ID COLUMN:", raw_id_col)

        matched_row = raw_df[
            raw_df[raw_id_col]
            .astype(str)
            .str.strip()
            .str.lower()
            == property_id
        ]

        if matched_row.empty:
            return {
                "success": False,
                "error": f"Original raw property not found for ID: {property_id}"
            }

        # ==========================================
        # USE FULL ORIGINAL ROW
        # SAME AS WORKING STREAMLIT PAGE
        # ==========================================
        raw_payload = (
            matched_row
            .iloc[0]
            .drop(labels=["PRICE"], errors="ignore")
            .to_dict()
        )

        payload = sanitize_payload(raw_payload)

        # print("\n========== PAYLOAD SIZE ==========")
        # print(len(payload))
        # print("==================================\n")

        # print("\n========== PAYLOAD ==========")
        # print(payload)
        # print("=============================\n")

        #This code is used to send your property data (payload) from your Streamlit/LangGraph app -
        #to the FastAPI prediction server so the ML model can predict the house price.
        response = requests.post(
            PREDICT_API_URL,
            json=payload,
            timeout=60
        )

        # print("STATUS:", response.status_code)
        # print("RESPONSE:", response.text)

        if response.status_code != 200:

            return {
                "success": False,
                "error": f"Prediction API failed ({response.status_code})"
            }

        data = response.json()

        return {
            "success": True,
            "prediction": data
        }
    
        

    except Exception as e:

        print("\n========== FULL ERROR ==========")
        traceback.print_exc()
        print("================================\n")

        # Handle FastAPI server connection errors.
        # Start server using:
        # python -m uvicorn app:app --reload --port 8000
        error_msg = str(e)

        # -----------------------------------------
        # FASTAPI SERVER NOT RUNNING
        # -----------------------------------------
        if (
            "127.0.0.1:8000" in error_msg
            or "Failed to establish a new connection" in error_msg
            or "actively refused" in error_msg
        ):

            return {
                "success": False,
                "error": (
                    "Prediction server is not running.\n\n"
                    "Please start the FastAPI prediction server using:\n"
                    "python -m uvicorn app:app --reload --port 8000"
                )
            }

        # -----------------------------------------
        # UNKNOWN ERROR
        # -----------------------------------------
        return {
            "success": False,
            "error": f"Prediction failed: {error_msg}"
        }





