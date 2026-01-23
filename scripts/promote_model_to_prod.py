# #promote_model_to_prod.py 2
# import mlflow
# import dagshub
# import json
# import os
# from mlflow import MlflowClient

# # Initialize DagsHub for your repository
# REPO_OWNER = 'bowlekarbhushan88'
# REPO_NAME = 'home-price-prediction'

# dagshub.init(
#     repo_owner=REPO_OWNER, 
#     repo_name=REPO_NAME, 
#     mlflow=True
# )

# # Set the MLflow tracking server URI
# TRACKING_URI = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
# mlflow.set_tracking_uri(TRACKING_URI)

# def load_model_information(file_path):
#     """Loads run and model metadata from the specified JSON file."""
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"Metadata file not found: {file_path}")
#     with open(file_path) as f:
#         return json.load(f)

# # Main Execution
# if __name__ == "__main__":
#     # 1. Load project metadata from your cache directory
#     RUN_INFO_PATH = "cache/run_information.json"
#     run_info = load_model_information(RUN_INFO_PATH)
    
#     model_name = run_info["model_name"]
#     source_stage = "staging"
#     target_stage = "production"
#     prod_alias = "champion" # Modern alias for Production

#     # 2. Initialize MLflow Client
#     client = MlflowClient()

#     # 3. Retrieve the latest version currently in Staging
#     latest_versions = client.get_latest_versions(name=model_name, stages=[source_stage])
    
#     if not latest_versions:
#         print(f"No model versions found in stage: {source_stage}. Skipping promotion.")
#     else:
#         latest_version_num = latest_versions[0].version
#         print(f"Promoting {model_name} v{latest_version_num} from {source_stage} to {target_stage}")

#         # 4. Transition the version to Production stage
#         client.transition_model_version_stage(
#             name=model_name,
#             version=latest_version_num,
#             stage=target_stage,
#             archive_existing_versions=True
#         )

#         # 5. Set the '@champion' alias for visibility in DagsHub UI
#         # This makes the version appear in the 'Aliased versions' column
#         client.set_registered_model_alias(
#             name=model_name,
#             alias=prod_alias,
#             version=latest_version_num
#         )
        
#         print(f"Successfully promoted to {target_stage} with alias '@{prod_alias}'.")


import json
import os
import mlflow
import dagshub
from mlflow import MlflowClient

# Constants
REPO_OWNER = "bowlekarbhushan88"
REPO_NAME = "home-price-prediction"

def load_model_information(file_path):
    """Loads run and model metadata from the specified JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metadata file not found: {file_path}")
    with open(file_path) as f:
        return json.load(f)

if __name__ == "__main__":
    # 1. Initialize DagsHub (Automatically handles authentication)
    dagshub.init(
        repo_owner=REPO_OWNER,
        repo_name=REPO_NAME,
        mlflow=True
    )

    # 2. Set the MLflow tracking server URI
    TRACKING_URI = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    # 3. Load project metadata
    RUN_INFO_PATH = "cache/run_information.json"
    run_info = load_model_information(RUN_INFO_PATH)
    model_name = run_info["model_name"]

    source_alias = "staging"
    target_alias = "production"

    # 4. Initialize MLflow Client
    client = MlflowClient()

    print(f"Attempting to promote model '{model_name}'...")

    try:
        # ✅ Get model version currently tagged as 'staging'
        staging_mv = client.get_model_version_by_alias(model_name, source_alias)
        version = staging_mv.version

        print(f"✅ Found version {version} with alias @{source_alias}")
        print(f"🚀 Promoting {model_name} version {version} -> @{target_alias}")

        # ✅ Set production alias to this version
        client.set_registered_model_alias(
            name=model_name,
            alias=target_alias,
            version=version
        )

        print(f"✅ Successfully promoted @{target_alias} to version {version}")

    except Exception as e:
        print(f"❌ Promotion failed: {e}")
        raise  # Ensure the GitHub Action fails if this script fails
