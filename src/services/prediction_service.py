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

# # ===============================
# # prediction_service.py
# # ===============================

# import requests
# import pandas as pd
# from pathlib import Path
# import sys
# import traceback
# import numpy as np

# ROOT_DIR = Path(__file__).resolve().parents[2]
# sys.path.append(str(ROOT_DIR))


# PREDICT_API_URL = "http://127.0.0.1:8000/predict"

# RAW_DATA_PATH = (
#     ROOT_DIR
#     / "data"
#     / "raw"
#     / "f_original magicbricks cleaned 12022 data.csv"
# )

# # ==========================================
# # SAFE JSON SANITIZER
# # ==========================================
# def sanitize_payload(row_dict):

#     clean = {}

#     for k, v in row_dict.items():

#         # ---------------------------------
#         # REMOVE NaN / None
#         # ---------------------------------
#         if pd.isna(v):
#             continue

#         # ---------------------------------
#         # REMOVE inf / -inf
#         # ---------------------------------
#         if isinstance(v, (float, np.floating)):

#             if np.isinf(v):
#                 continue

#         # ---------------------------------
#         # REMOVE list/dict
#         # ---------------------------------
#         if isinstance(v, (list, dict)):
#             continue

#         # ---------------------------------
#         # SAFE NUMERIC CAST
#         # ---------------------------------
#         if isinstance(v, (np.integer, int)):
#             clean[k] = int(v)

#         elif isinstance(v, (np.floating, float)):
#             clean[k] = float(v)

#         else:
#             clean[k] = str(v)

#     return clean



# def predict_property_price(property_row):

#     try:

#         # print("\n========== PROPERTY ROW ==========")
#         # print(property_row)
#         # print("=================================\n")

#         # print("\n========== PROPERTY ROW INDEX ==========")
#         # print(property_row.index.tolist())
#         # print("=======================================\n")

#         # ==========================================
#         # LOAD ORIGINAL RAW DATA
#         # ==========================================

#         raw_df = pd.read_csv(
#             RAW_DATA_PATH,
#             low_memory=False
#         )

#         # print("\n========== RAW DF COLUMNS ==========")
#         # print(raw_df.columns.tolist())
#         # print("===================================\n")

#         if "id" in property_row.index:
#             property_id = property_row["id"]

#         elif "ID" in property_row.index:
#             property_id = property_row["ID"]

#         else:
#             return {
#                 "success": False,
#                 "error": "No property ID column found"
#             }

#         property_id = str(property_id).strip().lower()


#         # ==========================================
#         # HANDLE RAW CSV ID COLUMN SAFELY
#         # ==========================================
#         raw_df.columns = raw_df.columns.str.strip()

#         # print("\n========== RAW DF CLEANED COLUMNS ==========")
#         # print(raw_df.columns.tolist())
#         # print("===========================================\n")

#         raw_id_col = None

#         if "id" in raw_df.columns:
#             raw_id_col = "id"

#         elif "ID" in raw_df.columns:
#             raw_id_col = "ID"

#         else:
#             return {
#                 "success": False,
#                 "error": f"No ID column found in raw CSV. Columns: {raw_df.columns.tolist()}"
#             }

#         # print("RAW ID COLUMN:", raw_id_col)

#         matched_row = raw_df[
#             raw_df[raw_id_col]
#             .astype(str)
#             .str.strip()
#             .str.lower()
#             == property_id
#         ]

#         if matched_row.empty:
#             return {
#                 "success": False,
#                 "error": f"Original raw property not found for ID: {property_id}"
#             }

#         # ==========================================
#         # USE FULL ORIGINAL ROW
#         # SAME AS WORKING STREAMLIT PAGE
#         # ==========================================
#         raw_payload = (
#             matched_row
#             .iloc[0]
#             .drop(labels=["PRICE"], errors="ignore")
#             .to_dict()
#         )

#         payload = sanitize_payload(raw_payload)

#         # print("\n========== PAYLOAD SIZE ==========")
#         # print(len(payload))
#         # print("==================================\n")

#         # print("\n========== PAYLOAD ==========")
#         # print(payload)
#         # print("=============================\n")

#         #This code is used to send your property data (payload) from your Streamlit/LangGraph app -
#         #to the FastAPI prediction server so the ML model can predict the house price.
#         response = requests.post(
#             PREDICT_API_URL,
#             json=payload,
#             timeout=60
#         )

#         # print("STATUS:", response.status_code)
#         # print("RESPONSE:", response.text)

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

#         print("\n========== FULL ERROR ==========")
#         traceback.print_exc()
#         print("================================\n")

#         # Handle FastAPI server connection errors.
#         # Start server using:
#         # python -m uvicorn app:app --reload --port 8000
#         error_msg = str(e)

#         # -----------------------------------------
#         # FASTAPI SERVER NOT RUNNING
#         # -----------------------------------------
#         if (
#             "127.0.0.1:8000" in error_msg
#             or "Failed to establish a new connection" in error_msg
#             or "actively refused" in error_msg
#         ):

#             return {
#                 "success": False,
#                 "error": (
#                     "Prediction server is not running.\n\n"
#                     "Please start the FastAPI prediction server using:\n"
#                     "python -m uvicorn app:app --reload --port 8000"
#                 )
#             }

#         # -----------------------------------------
#         # UNKNOWN ERROR
#         # -----------------------------------------
#         return {
#             "success": False,
#             "error": f"Prediction failed: {error_msg}"
#         }


#===================================================================================================================================================================================



# ==========================================
# prediction_service.py
# ==========================================

import sys
from pathlib import Path
import traceback
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# ------------------------------------------
# PATH & SYSTEM PATH CONFIGURATION
# ------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# ==========================================
# CONFIGURATION & GLOBAL STATE
# ==========================================
ARTIFACTS_DIR = ROOT_DIR / "cache" / "artifacts_auto"
TE_COLS = ["builder", "project_name", "location"]

# Artifact mapping: { Logical Name: Actual Filename in MLflow }
REQUIRED_ARTIFACTS = {
    "cleaner": "cleaner_full_for_te.pkl",
    "te": "te_full.pkl",
    "preprocessor": "preprocessor.joblib",
    "model": "final_model.joblib"
}

# Persistent in-memory cache for app lifecycle
_CACHED_ARTIFACTS = None


def download_explicit_artifacts(run_id):
    """
    Downloads each artifact explicitly by name from MLflow to guarantee
    they are placed directly into ARTIFACTS_DIR without nested run folders.
    """
    import mlflow
    
    print(f"📦 Fetching 4 explicit components from Run ID: {run_id}")
    for log_name, file_name in REQUIRED_ARTIFACTS.items():
        print(f"       -> Downloading {file_name}...")
        
        # Explicitly pull the singular file asset
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=file_name,
            dst_path=str(ARTIFACTS_DIR)
        )
    print("✅ All explicit components downloaded and cached locally!")


def get_or_download_artifacts():
    """
    Returns loaded components. If missing from disk, queries the 'DVC Pipeline'
    experiment on MLflow, resolves the latest run, and pulls them down.
    """
    global _CACHED_ARTIFACTS
    if _CACHED_ARTIFACTS is not None:
        return _CACHED_ARTIFACTS

    # Ensure local layout target exists
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve exact local paths
    local_paths = {name: ARTIFACTS_DIR / filename for name, filename in REQUIRED_ARTIFACTS.items()}
    missing_artifacts = [name for name, path in local_paths.items() if not path.exists()]

    if missing_artifacts:
        print(f"⚠️ Missing local files: {missing_artifacts}.")
        print("🕒 Cold start: Querying MLflow/DagsHub. First prediction will take 10-30s...")
        try:
            import mlflow
            
            # Fix Problem 2: Point directly to 'DVC Pipeline' matching evaluation.py
            EXPERIMENT_NAME = "DVC Pipeline"
            experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            
            if experiment is None:
                raise ValueError(f"MLflow experiment '{EXPERIMENT_NAME}' not found. Check DagsHub UI.")
            
            # Fetch the latest successful run belonging to this experiment
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="status = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if runs.empty:
                raise ValueError(f"No finished runs discovered inside experiment '{EXPERIMENT_NAME}'.")
                
            latest_run_id = runs.iloc[0]["run_id"]
            
            # Fix Problem 1: Direct targeted downloads
            download_explicit_artifacts(latest_run_id)

        except Exception as err:
            print(f"❌ Critical failure while pulling artifacts from MLflow: {str(err)}")
            raise err

    # Load components safely into memory mapping
    print("🧠 Initializing model components into memory map...")
    _CACHED_ARTIFACTS = {name: joblib.load(path) for name, path in local_paths.items()}
    
    # Secure unified prediction pipeline
    _CACHED_ARTIFACTS["model_pipe"] = Pipeline([
        ("preprocessor", _CACHED_ARTIFACTS["preprocessor"]),
        ("model", _CACHED_ARTIFACTS["model"])
    ])

    return _CACHED_ARTIFACTS


# ==========================================
# SAFE JSON SANITIZER
# ==========================================
def sanitize_payload(row_dict):
    clean = {}
    for k, v in row_dict.items():
        if pd.isna(v):
            continue
        if isinstance(v, (float, np.floating)) and np.isinf(v):
            continue
        if isinstance(v, (list, dict)):
            continue

        if isinstance(v, (np.integer, int)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            clean[k] = float(v)
        else:
            clean[k] = str(v)
    return clean


# ==========================================
# TARGET ENCODING HELPER
# ==========================================
def apply_te(df, artifacts):
    df = df.copy()

    for col in TE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    X_clean = artifacts["cleaner"].transform(df[TE_COLS].copy())
    X_te_df = artifacts["te"].transform(X_clean)

    for col in TE_COLS:
        df[f"{col}_te"] = X_te_df[col].values

    df.drop(columns=TE_COLS, inplace=True, errors="ignore")
    return df


# ==========================================
# DIRECT LOCAL PREDICTION SERVICE
# ==========================================
def predict_property_price(property_row):
    try:
        print("\n🚀 DIRECT MODEL PREDICTION RUNNING")
        
        # Lazy initialization happens here at runtime
        artifacts = get_or_download_artifacts()

        raw_payload = property_row.to_dict()
        payload = sanitize_payload(raw_payload)

        X = pd.DataFrame([payload])
        X = apply_te(X, artifacts)

        pred_log = artifacts["model_pipe"].predict(X)
        predicted_price = float(np.expm1(pred_log)[0])

        return {
            "success": True,
            "prediction": {
                "predicted_price": round(predicted_price, 2),
                "unit": "Cr"
            }
        }

    except Exception as e:
        print("\n========== ERROR LOGGED ==========")
        traceback.print_exc()
        print("==================================\n")
        return {
            "success": False,
            "error": str(e)
        }