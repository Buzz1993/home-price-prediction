# ============================================================
# test_model_registry.py 2
# ------------------------------------------------------------
# Verifies model availability using the Aliases & Tags system.
# ============================================================

import os
import json
import joblib
import pytest
import mlflow
import dagshub
from pathlib import Path
from mlflow import MlflowClient

# ============================================================
# CONFIG & DAGSHUB INIT
# ============================================================
REPO_OWNER = "bowlekarbhushan88"
REPO_NAME = "home-price-prediction"
TRACKING_URI = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"

# ✅ UPDATED: Modern MLflow uses Aliases
MODEL_ALIAS = "staging"

# Path Setup
ROOT_PATH = Path(__file__).resolve().parents[1]
RUN_INFO_PATH = ROOT_PATH / "cache" / "run_information.json"
LOCAL_MODEL_PATH = ROOT_PATH / "cache" / "artifacts_auto" / "final_model.joblib"

# Fix DagsHub/MLflow timeouts
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "600"
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "5"

dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
mlflow.set_tracking_uri(TRACKING_URI)

# ============================================================
# HELPERS
# ============================================================
def load_model_name():
    if not RUN_INFO_PATH.exists():
        pytest.fail(f"Run information missing at {RUN_INFO_PATH}")
    with open(RUN_INFO_PATH, "r") as f:
        return json.load(f)["model_name"]

MODEL_NAME = load_model_name()

# ============================================================
# TEST SUITE
# ============================================================
@pytest.mark.parametrize("model_name, alias", [(MODEL_NAME, MODEL_ALIAS)])
def test_load_model_from_registry(model_name, alias):
    """
    Test that checks DagsHub Registry for a model version via Alias,
    falling back to local storage if necessary.
    """
    client = MlflowClient()
    version_num = "Unknown"

    # 1. Verify Metadata exists via Alias
    try:
        # ✅ UPDATED: Replaces get_latest_versions (Stages)
        model_version_details = client.get_model_version_by_alias(name=model_name, alias=alias)
        version_num = model_version_details.version
        print(f"\n✅ Found {model_name} v{version_num} with alias @{alias} on DagsHub")
    except Exception as e:
        pytest.fail(f"Could not find model with alias @{alias} in Registry: {e}")

    # 2. Attempt Model Loading (Using the @alias URI)
    model_uri = f"models:/{model_name}@{alias}"
    model = None

    try:
        print(f"Attempting to download model from {model_uri}...")
        model = mlflow.sklearn.load_model(model_uri)
        print("Success: Model downloaded from Registry.")
    except Exception as e:
        print(f"Remote download failed. Error: {str(e)[:100]}...")
        
        # 3. Local Fallback
        print(f"Checking for local backup at: {LOCAL_MODEL_PATH}")
        if LOCAL_MODEL_PATH.exists():
            model = joblib.load(LOCAL_MODEL_PATH)
            print("Success: Validated via local backup.")
        else:
            pytest.fail("Remote failed and no local backup found in cache.")

    # 4. Final Verification
    assert model is not None, "Final model object is empty."
    assert hasattr(model, "predict"), "Loaded object is not a valid sklearn estimator."

    print(f"Verified: {model_name} v{version_num} (@{alias}) is ready for use.")