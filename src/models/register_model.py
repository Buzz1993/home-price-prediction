
# ============================================================
# register_model.py
# ------------------------------------------------------------
# Registers the latest trained MLflow model into DagsHub Model Registry
# and moves it to "Staging" stage.
#
# Reads:
#   cache/run_information.json
# which is created at the end of evaluation.py
# ============================================================

import json
import logging
from pathlib import Path

import mlflow
import dagshub
from mlflow import MlflowClient


# =====================================================
# LOGGER SETUP
# =====================================================
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)


# =====================================================
# DAGSHUB + MLFLOW SETUP
# =====================================================
dagshub.init(
    repo_owner="bowlekarbhushan88",
    repo_name="home-price-prediction",
    mlflow=True
)

TRACKING_URI = "https://dagshub.com/bowlekarbhushan88/home-price-prediction.mlflow"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)


# =====================================================
# HELPERS
# =====================================================
def load_model_information(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"run_information.json not found at: {file_path}")

    with open(file_path, "r") as f:
        run_info = json.load(f)

    required_keys = ["run_id", "model_name", "artifact_path"]
    missing = [k for k in required_keys if k not in run_info]
    if missing:
        raise KeyError(f"Missing keys in run_information.json: {missing}")

    return run_info

from mlflow.exceptions import RestException

def ensure_registered_model_exists(client: MlflowClient, model_name: str):
    """
    Create registered model if it doesn't exist (needed for first time on DagsHub).
    """
    try:
        client.get_registered_model(model_name)
        logger.info(f"✅ Registered model already exists: {model_name}")
    except RestException as e:
        # model not found
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            logger.info(f"Registered model not found. Creating: {model_name}")
            client.create_registered_model(model_name)
            logger.info(f"✅ Created registered model: {model_name}")
        else:
            raise



# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    root_path = Path(__file__).parent.parent.parent

    run_info_path = root_path / "cache" / "run_information.json"
    run_info = load_model_information(run_info_path)

    run_id = run_info["run_id"]
    model_name = run_info["model_name"]
    artifact_path = run_info["artifact_path"]

    logger.info(f"Loaded run info | run_id={run_id}, model_name={model_name}, artifact_path={artifact_path}")

    # Register model using MlflowClient
    client = MlflowClient()

    # =====================================================
    # ✅ IMPORTANT FIX (DagsHub registry issue)
    # Instead of source="runs:/<run_id>/<artifact_path>"
    # Use absolute artifact URI from the run
    # =====================================================
    run_obj = client.get_run(run_id)
    artifact_uri = run_obj.info.artifact_uri

    # Absolute model source path
    model_source = f"{artifact_uri}/{artifact_path}"

    logger.info(f"Resolved Run Artifact URI : {artifact_uri}")
    logger.info(f"Registering model from    : {model_source}")

    # ✅ ensure model exists in registry
    ensure_registered_model_exists(client, model_name)

    logger.info("Creating a new model version in registry...")
    model_version_obj = client.create_model_version(
        name=model_name,
        source=model_source,
        run_id=run_id
    )


    registered_model_version = model_version_obj.version
    registered_model_name = model_version_obj.name

    logger.info(
        f"Model registered successfully | Name={registered_model_name}, Version={registered_model_version}"
    )

    # Transition to Staging
    client.transition_model_version_stage(
        name=registered_model_name,
        version=registered_model_version,
        stage="Staging"
    )

    logger.info(f"Model pushed to Staging stage | {registered_model_name} v{registered_model_version}")

    # Save registry info to local file
    out_dir = root_path / "cache" / "model_registry"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "registered_model_info.json"

    with open(out_path, "w") as f:
        json.dump(
            {
                "model_name": registered_model_name,
                "model_version": registered_model_version,
                "stage": "Staging",
                "run_id": run_id,
                "model_source": model_source
            },
            f,
            indent=4
        )

    logger.info(f"Saved registry info to: {out_path}")

