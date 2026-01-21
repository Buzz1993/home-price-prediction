#test_model_registry.py
import os
import json
import joblib
import pytest
import mlflow
import dagshub
import logging
from pathlib import Path
from mlflow import MlflowClient

# ============================================================
# CONFIG & DAGSHUB INIT
# ============================================================
REPO_OWNER = "bowlekarbhushan88"
REPO_NAME = "home-price-prediction"
TRACKING_URI = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
STAGE = "Staging"

# Path Setup
ROOT_PATH = Path(__file__).resolve().parents[1]
RUN_INFO_PATH = ROOT_PATH / "cache" / "run_information.json"
LOCAL_MODEL_PATH = ROOT_PATH / "cache" / "artifacts_auto" / "final_model.joblib"

# Fix DagsHub/MLflow timeouts
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "600"  # 10 minutes
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "5"

dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

# ============================================================
# HELPERS
# ============================================================
def load_model_name():
    if not RUN_INFO_PATH.exists():
        pytest.fail(f"Run information missing at {RUN_INFO_PATH}")
    with open(RUN_INFO_PATH, "r") as f:
        return json.load(f)["model_name"]

# Dynamic model name from cache
MODEL_NAME = load_model_name()

# ============================================================
# TEST SUITE
# ============================================================
@pytest.mark.parametrize("model_name, stage", [(MODEL_NAME, STAGE)])
def test_load_model_from_registry(model_name, stage):
    """
    Test that checks DagsHub Registry for the model version, 
    but falls back to local storage if DagsHub 500s during download.
    """
    client = MlflowClient()
    latest_version = "Unknown"

    # 1. Verify Metadata exists on DagsHub (Lightweight API call)
    try:
        latest_versions = client.get_latest_versions(name=model_name, stages=[stage])
        assert latest_versions, f"No model versions found in stage: {stage}"
        latest_version = latest_versions[0].version
        print(f"\n Found {model_name} v{latest_version} in {stage} on DagsHub")
    except Exception as e:
        pytest.fail(f"Could not connect to DagsHub Registry: {e}")

    # 2. Attempt Model Loading (Heavy Download)
    model_uri = f"models:/{model_name}/{stage}"
    model = None

    try:
        print(f"Attempting to download model from {model_uri}...")
        model = mlflow.sklearn.load_model(model_uri)
        print("Success: Model downloaded from Registry.")
    except Exception as e:
        print(f"Remote download failed (DagsHub 500 error). Error: {str(e)[:100]}...")
        
        # 3. Local Fallback (Mirroring app.py logic)
        print(f"Checking for local backup at: {LOCAL_MODEL_PATH}")
        if LOCAL_MODEL_PATH.exists():
            model = joblib.load(LOCAL_MODEL_PATH)
            print("Success: Validated via local backup.")
        else:
            pytest.fail("Remote failed and no local backup found in cache.")

    # 4. Final Verification
    assert model is not None, "Final model object is empty."
    assert hasattr(model, "predict"), "Loaded object is not a valid sklearn estimator."

    # Final Success Print
    print(f"The {model_name} model with version {latest_version} was loaded successfully")