# ===============================
# check_artifacts.py
# ===============================

import mlflow, dagshub, os
from pathlib import Path

os.environ["MLFLOW_TRACKING_USERNAME"] = "bowlekarbhushan88"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN") #this token get from .env file 

dagshub.init(repo_owner="bowlekarbhushan88", repo_name="home-price-prediction", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/bowlekarbhushan88/home-price-prediction.mlflow")
mlflow.set_experiment("debug")

with mlflow.start_run():
    Path("tmp.txt").write_text("hello")
    mlflow.log_artifact("tmp.txt")
    print("done")
