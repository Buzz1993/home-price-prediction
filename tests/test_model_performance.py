#test_model_performance.py
import os
import json
import joblib
import pytest
import mlflow
import dagshub
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from mlflow import MlflowClient

from sklearn import set_config
set_config(transform_output="pandas")

# ============================================================
# CONFIG & DAGSHUB INIT
# ============================================================
REPO_OWNER = "bowlekarbhushan88"
REPO_NAME = "home-price-prediction"
TRACKING_URI = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
STAGE = "Staging"

# Thresholds
THRESHOLD_MAE = 0.50   # 50 Lakhs
THRESHOLD_MAPE = 20.0  # 20% average error

# Path Setup
ROOT_PATH = Path(__file__).resolve().parents[1]
RUN_INFO_PATH = ROOT_PATH / "cache" / "run_information.json"
ARTIFACTS_DIR = ROOT_PATH / "cache" / "artifacts_auto"
TEST_DATA_PATH = ROOT_PATH / "data" / "interim" / "test.csv"

# Fix DagsHub/MLflow timeouts
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "600"
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "5"

dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

# ============================================================
# LOAD METADATA & ARTIFACTS
# ============================================================
def load_run_info():
    with open(RUN_INFO_PATH, "r") as f:
        return json.load(f)

run_info = load_run_info()
MODEL_NAME = run_info["model_name"]
TARGET_COL = "price"

# Load local supporting artifacts (Cleaner, TE, Preprocessor)
cleaner = joblib.load(ARTIFACTS_DIR / "cleaner_full_for_te.pkl")
te_full = joblib.load(ARTIFACTS_DIR / "te_full.pkl")
preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")

# ============================================================
# HELPER: LOAD MODEL (With Fallback)
# ============================================================
def get_model():
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    try:
        print(f"Attempting to load model from Registry: {model_uri}")
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Remote load failed (DagsHub 500), using local backup.")
        return joblib.load(ARTIFACTS_DIR / "final_model.joblib")

# ============================================================
# PERFORMANCE TEST
# ============================================================
def test_model_performance():
    """
    Evaluates the model on test data using MAE and MAPE thresholds.
    """
    # 1. Load Model
    model = get_model()
    
    # 2. Build Pipeline (Preprocessor + Estimator)
    model_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    # 3. Load Test Data
    if not TEST_DATA_PATH.exists():
        pytest.fail(f"Test data not found at {TEST_DATA_PATH}")
        
    df = pd.read_csv(TEST_DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y_true = df[TARGET_COL]

    # 4. Apply Target Encoding (Must match app.py logic)
    X_te_cols = ["builder", "project_name", "location"]
    X_clean = cleaner.transform(X[X_te_cols].copy())
    X_te_df = te_full.transform(X_clean)
    
    X_prepared = X.copy()
    for col in X_te_cols:
        X_prepared[f"{col}_te"] = X_te_df[col].values
    X_prepared.drop(columns=X_te_cols, inplace=True, errors="ignore")

    # 5. Get Predictions (Log Scale) and Inverse Transform
    y_pred_log = model_pipe.predict(X_prepared)
    y_pred = np.expm1(y_pred_log)

    # 6. Calculate Metrics
    mae = mean_absolute_error(y_true, y_pred)
    # MAPE calculation (multiplying by 100 for percentage)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    print(f"\n--- Performance Summary ---")
    print(f"MAE : {mae:.4f} Cr")
    print(f"MAPE: {mape:.2f} %")
    
    # 7. Assertions
    assert mae <= THRESHOLD_MAE, f"Model MAE {mae:.4f} exceeds threshold {THRESHOLD_MAE} Cr"
    assert mape <= THRESHOLD_MAPE, f"Model MAPE {mape:.2f}% exceeds threshold {THRESHOLD_MAPE}%"
    
    print(f"\nThe {MODEL_NAME} model passed all performance tests!")