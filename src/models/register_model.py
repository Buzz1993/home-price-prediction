# ============================================================
# register_model.py 2
# ------------------------------------------------------------
# Registers model using the modern Aliases & Tags system.
# ============================================================

import json
import logging
from pathlib import Path

import mlflow
import dagshub
from mlflow import MlflowClient
from mlflow.exceptions import RestException

# =====================================================
# LOGGER SETUP
# =====================================================
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)

# =====================================================
# DAGSHUB + MLFLOW SETUP
# =====================================================
dagshub.init(repo_owner="bowlekarbhushan88", repo_name="home-price-prediction", mlflow=True)
TRACKING_URI = "https://dagshub.com/bowlekarbhushan88/home-price-prediction.mlflow"
mlflow.set_tracking_uri(TRACKING_URI)

# =====================================================
# HELPERS
# =====================================================
def load_model_information(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"run_information.json not found at: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)

def ensure_registered_model_exists(client: MlflowClient, model_name: str):
    try:
        client.get_registered_model(model_name)
    except RestException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            logger.info(f"Creating new registered model: {model_name}")
            client.create_registered_model(model_name)
        else:
            raise

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    root_path = Path(__file__).parent.parent.parent
    run_info = load_model_information(root_path / "cache" / "run_information.json")

    run_id = run_info["run_id"]
    model_name = run_info["model_name"]
    artifact_path = run_info["artifact_path"]

    client = MlflowClient()

    # Get absolute URI for DagsHub compatibility
    run_obj = client.get_run(run_id)
    model_source = f"{run_obj.info.artifact_uri}/{artifact_path}"

    ensure_registered_model_exists(client, model_name)

    logger.info(f"Registering version for {model_name}...")
    mv = client.create_model_version(
        name=model_name,
        source=model_source,
        run_id=run_id
    )

    # =====================================================
    # ✅ MODERN MLFLOW SYSTEM: ALIASES & TAGS
    # =====================================================
    
    # 1. Set an Alias (This replaces "Stage")
    # This will show up in the "Aliased versions" column in the UI
    client.set_registered_model_alias(
        name=model_name,
        alias="staging",
        version=mv.version
    )
    logger.info(f"✅ Set alias 'staging' for version {mv.version}")

    # 2. Set Tags (For metadata visibility)
    # Tags are visible in the "Tags" section of the version details
    client.set_model_version_tag(
        name=model_name,
        version=mv.version,
        key="deployment_status",
        value="pending_review"
    )
    
    # =====================================================

    # Save registry info to local file (Updated for Aliases)
    out_dir = root_path / "cache" / "model_registry"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "registered_model_info.json", "w") as f:
        json.dump({
            "model_name": mv.name,
            "model_version": mv.version,
            "alias": "staging",  # Replaced 'stage'
            "run_id": run_id,
            "model_source": model_source
        }, f, indent=4)

    logger.info(f"Successfully registered model version {mv.version} with alias '@staging'")