#check_artifacts.py
# import mlflow
# import dagshub
# from mlflow import MlflowClient

# dagshub.init(repo_owner="bowlekarbhushan88", repo_name="home-price-prediction", mlflow=True)
# mlflow.set_tracking_uri("https://dagshub.com/bowlekarbhushan88/home-price-prediction.mlflow")

# run_id = "a63bf4d777d5443f98e941a3d0358495"

# client = MlflowClient()
# files = client.list_artifacts(run_id)

# print("Artifacts in run:", run_id)
# for f in files:
#     print("-", f.path)


import mlflow, dagshub, os
from pathlib import Path

os.environ["MLFLOW_TRACKING_USERNAME"] = "bowlekarbhushan88"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

dagshub.init(repo_owner="bowlekarbhushan88", repo_name="home-price-prediction", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/bowlekarbhushan88/home-price-prediction.mlflow")
mlflow.set_experiment("debug")

with mlflow.start_run():
    Path("tmp.txt").write_text("hello")
    mlflow.log_artifact("tmp.txt")
    print("done")
