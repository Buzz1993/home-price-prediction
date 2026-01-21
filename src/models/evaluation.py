# ============================================================
# evaluation.py  (SWIGGY STYLE)
# ------------------------------------------------------------
# Stage 3 + Evaluation in ONE script:
#
# Stage 3:
# - Load Stage1 artifact best_trial_params_auto.pkl
# - Train FULL Target Encoding artifacts
# - Fit preprocessor on full train
# - Train FINAL estimator ONLY (no full pipeline in registry)
# - Save locally:
#    - preprocessor.joblib
#    - cleaner_full_for_te.pkl
#    - te_full.pkl
#    - final_model.joblib  (optional local backup)
#
# Evaluation:
# - Load train/test
# - Apply TE
# - Apply preprocessor
# - Predict using final_model
# - Compute metrics Train & Test
# - Save metrics + predictions
# - Log ONLY estimator model to MLflow (DagsHub)
# ============================================================

import os
import json
import yaml
import joblib
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn import set_config
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from category_encoders.target_encoder import TargetEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import mlflow
import dagshub


# ============================================================
# GLOBAL CONFIG
# ============================================================
set_config(transform_output="pandas")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


# ============================================================
# DAGSHUB + MLFLOW SETUP
# ============================================================
dagshub.init(
    repo_owner="bowlekarbhushan88",
    repo_name="home-price-prediction",
    mlflow=True
)

mlflow.set_tracking_uri("https://dagshub.com/bowlekarbhushan88/home-price-prediction.mlflow")
mlflow.set_experiment("DVC Pipeline")


# ============================================================
# LOGGER
# ============================================================
logger = logging.getLogger("evaluation")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)


# ============================================================
# HELPERS
# ============================================================
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.where(y_true == 0, 1e-8, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def calc_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE": mape}


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def read_params(file_path: Path) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def ensure_cols_exist(df: pd.DataFrame, cols: list, name="data"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {name}: {missing}")


# ============================================================
# TRANSFORMERS
# ============================================================
from src.models.custom_transformers import (
    MultiLabelBinarizerTransformer,
    FrequencyThresholdCategoriesTransformer,
    MissingIndicatorAdder,
    ImputeAndScaleAmenity,
    ConstructionOrdinalEncoder,
    CategoryCleaner,
)

# ============================================================
# SAME COLUMN SETUP (from preprocess_train)
# ============================================================
KNN_NEIGH_DEFAULT = 5

te_cols = ["builder", "project_name", "location"]
freq_thresholds = {"project_name": 50, "builder": 20, "location": 20}

features_to_fill_knn = [
    'project_in_acres','area','education_mean_km','education_min_km',
    'transport_mean_km','transport_min_km','shopping_centre_mean_km','shopping_centre_min_km',
    'overall_min_mean_km','overall_avg_mean_km','overall_min_min_km','overall_avg_min_km',
    'available_units','towers','flat_on_floor','total_floor','bath','parking',
    'commercial_hub_mean_km','commercial_hub_min_km','balcony','bath_bed_ratio','bed_area_ratio',
    'bed_bath_ratio','bed_balcony_ratio','project_density','compactness_ratio','floor_ratio','remaining_floors',
    'area_per_bedroom','area_per_bathroom','area_per_balcony','area_per_parking','balcony_to_bed_ratio',
    'parking_to_bed_ratio','lift_to_total_floor_ratio',"water_availability_hours","locality_rank",
    "locality_rating","locality_review_count","environment_rating","commuting_rating",
    "places_of_interest_rating","project_age_months"
]
features_to_fill_iterative = ['lattitude','longitude']
features_to_fill_median = ['lift']

impute_mf_and_OHE = ['property_type']
impute_mf_and_ordinal_encode = ['status','furnish']
ordinal_categories = [['under construction','ongoing','ready to move'],
                      ['unfurnished','semi-furnished','furnished']]

impute_missing_and_OHE = ['ownership','facing']
impute_missing_and_multilable = ['overlooking','extra_rooms','flooring']

assignweight_missingindicator_KNNimputation_minmaxscale = ['assigned_amenities_score']

missingindicator_ordinal_encode = ['construction']
construction_categories = [[
    'missing','under construction','new construction','less than 5 years',
    '5 to 10 years','10 to 15 years','15 to 20 years','above 20 years'
]]

onehotencode = ['city','seller']

min_max_scaling = [
    'education_within_2km','transport_within_2km','shopping_centre_within_2km',
    'commercial_hub_within_2km','hospital_within_2km','tourist_within_2km','total_within_2km'
]
knn_impute_scaling = ['distance_to_center_km']


# ============================================================
# PIPELINES
# ============================================================
builder_location_project_name_pipeline = Pipeline([
    ("cat_clean", CategoryCleaner(cols=te_cols)),
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("clip", FrequencyThresholdCategoriesTransformer(thresholds=freq_thresholds)),
])

property_type_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy="most_frequent")),
    ('OHE', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

status_furnish_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy="most_frequent")),
    ('ordinal_encode', OrdinalEncoder(
        categories=ordinal_categories,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    ))
])

ownership_facing_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy="constant", fill_value="missing")),
    ('OHE', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

overlooking_extra_rooms_flooring_pipeline = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
    ("multilable", MultiLabelBinarizerTransformer())   
])


assigned_amenities_pipeline = Pipeline([
    ('add_missing_indicator', MissingIndicatorAdder(column='assigned_amenities_score')),
    ('impute_and_scale', ImputeAndScaleAmenity(column='assigned_amenities_score', n_neighbors=KNN_NEIGH_DEFAULT))
])

construction_pipeline = Pipeline([
    ('add_missing_indicator', MissingIndicatorAdder(column='construction')),
    ('ordinal_encode', ConstructionOrdinalEncoder(column='construction', categories=construction_categories))
])

city_seller_pipeline = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

distance_to_center_km_pipeline = Pipeline([
    ('impute', KNNImputer(n_neighbors=KNN_NEIGH_DEFAULT)),
    ('scaler', MinMaxScaler())
])

within2km_pipeline = Pipeline([("scaler", StandardScaler())])

preprocessor = make_column_transformer(
    (KNNImputer(n_neighbors=KNN_NEIGH_DEFAULT), features_to_fill_knn),
    (SimpleImputer(strategy="median"), features_to_fill_iterative),
    (SimpleImputer(strategy="median"), features_to_fill_median),
    (property_type_pipeline, impute_mf_and_OHE),
    (status_furnish_pipeline, impute_mf_and_ordinal_encode),
    (ownership_facing_pipeline, impute_missing_and_OHE),
    (overlooking_extra_rooms_flooring_pipeline, impute_missing_and_multilable),
    (assigned_amenities_pipeline, assignweight_missingindicator_KNNimputation_minmaxscale),
    (construction_pipeline, missingindicator_ordinal_encode),
    (city_seller_pipeline, onehotencode),
    (distance_to_center_km_pipeline, knn_impute_scaling),
    (within2km_pipeline, min_max_scaling),
    remainder=SimpleImputer(strategy="median"),
    verbose_feature_names_out=False,
    n_jobs=1
)


# ============================================================
# MODEL BUILD
# ============================================================
def build_final_estimator(best_model_name: str, best_params: dict, random_state: int = 42):
    if best_model_name == "LGBM":
        return LGBMRegressor(**best_params, random_state=random_state, n_jobs=1, verbosity=-1)
    elif best_model_name == "RF":
        return RandomForestRegressor(**best_params, random_state=random_state, n_jobs=1)
    elif best_model_name == "ET":
        return ExtraTreesRegressor(**best_params, random_state=random_state, n_jobs=1)
    elif best_model_name == "XGB":
        return XGBRegressor(**best_params, random_state=random_state, n_jobs=1, verbosity=0, tree_method="hist")
    else:
        raise ValueError(f"Unknown model: {best_model_name}")


def apply_te_with_artifacts(X: pd.DataFrame, cleaner_full, te_full, te_cols_: list):
    X = X.copy()
    X_clean = cleaner_full.transform(X[te_cols_].copy())
    X_te_df = te_full.transform(X_clean)

    for col in te_cols_:
        X[f"{col}_te"] = X_te_df[col].values

    X.drop(columns=te_cols_, inplace=True, errors="ignore")
    return X


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    root_path = Path(__file__).parent.parent.parent

    params = read_params(root_path / "params.yaml")
    train_params = params.get("Train", {})
    TARGET = train_params.get("target_col", "price")

    logger.info(f"TARGET from params.yaml: {TARGET}")

    train_path = root_path / "data" / "interim" / "train.csv"
    test_path = root_path / "data" / "interim" / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if TARGET not in train_df.columns:
        raise ValueError(f"TARGET='{TARGET}' not found in train.csv")

    X_train = train_df.drop(columns=[TARGET]).copy()
    y_train = train_df[TARGET].copy()

    X_test = test_df.drop(columns=[TARGET], errors="ignore").copy()
    y_test = test_df[TARGET].copy() if TARGET in test_df.columns else None

    artifacts_dir = root_path / "cache" / "artifacts_auto"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = root_path / "cache" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    run_info_path = root_path / "cache" / "run_information.json"

    stage1_best_params_path = artifacts_dir / "best_trial_params_auto.pkl"

    # NEW artifacts
    preprocessor_path = artifacts_dir / "preprocessor.joblib"
    model_local_path = artifacts_dir / "final_model.joblib"
    cleaner_path = artifacts_dir / "cleaner_full_for_te.pkl"
    te_path = artifacts_dir / "te_full.pkl"

    if not stage1_best_params_path.exists():
        raise FileNotFoundError(f"Stage1 artifact not found: {stage1_best_params_path}")

    # ============================================================
    # STAGE 3 — ALWAYS REBUILD (recommended)
    # ============================================================
    logger.info("Building Stage3 artifacts (Swiggy style)...")

    best_trial_params = joblib.load(stage1_best_params_path)
    best_model_name = best_trial_params["model"]
    best_params = {k: v for k, v in best_trial_params.items() if k != "model"}

    # Fit TE artifacts
    ensure_cols_exist(X_train, te_cols, name="X_train")

    cleaner_full_for_te = clone(builder_location_project_name_pipeline)
    cleaner_full_for_te.fit(X_train[te_cols])

    y_train_log = np.log1p(y_train)
    X_train_clean = cleaner_full_for_te.transform(X_train[te_cols])

    te_full = TargetEncoder(smoothing=0.5)
    te_full.fit(X_train_clean, y_train_log)

    # Apply TE
    X_train_stage3 = apply_te_with_artifacts(X_train, cleaner_full_for_te, te_full, te_cols)
    X_test_stage3 = apply_te_with_artifacts(X_test, cleaner_full_for_te, te_full, te_cols)

    # Fit PREPROCESSOR
    preproc = clone(preprocessor)
    preproc.fit(X_train_stage3, y_train_log)

    X_train_pre = preproc.transform(X_train_stage3)
    X_test_pre = preproc.transform(X_test_stage3)

    # Train MODEL ONLY
    final_model = build_final_estimator(best_model_name, best_params, random_state=42)
    final_model.fit(X_train_pre, y_train_log)

    # Save local artifacts
    joblib.dump(preproc, preprocessor_path)
    joblib.dump(final_model, model_local_path)
    joblib.dump(cleaner_full_for_te, cleaner_path)
    joblib.dump(te_full, te_path)

    logger.info("Saved Swiggy-style artifacts:")
    logger.info(f"- {preprocessor_path.name}")
    logger.info(f"- {model_local_path.name}")
    logger.info(f"- {cleaner_path.name}")
    logger.info(f"- {te_path.name}")

    # ============================================================
    # EVALUATION
    # ============================================================
    y_train_pred_log = final_model.predict(X_train_pre)
    y_train_pred = np.expm1(y_train_pred_log)

    train_metrics = calc_metrics(y_train, y_train_pred)
    logger.info(f"TRAIN metrics: {train_metrics}")

    y_test_pred_log = final_model.predict(X_test_pre)
    y_test_pred = np.expm1(y_test_pred_log)

    test_metrics = None
    if y_test is not None:
        test_metrics = calc_metrics(y_test, y_test_pred)
        logger.info(f"TEST metrics: {test_metrics}")
    else:
        logger.warning("No target column found in test.csv → skipping test metrics")

    # save metrics
    result = {
        "target_col": TARGET,
        "best_model_name": best_model_name,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    metrics_json_path = eval_dir / "eval_metrics.json"
    save_json(result, metrics_json_path)

    # save predictions
    train_pred_path = eval_dir / "predictions_train.csv"
    pd.DataFrame({"y_true": y_train, "y_pred": y_train_pred}).to_csv(train_pred_path, index=False)

    test_pred_path = eval_dir / "predictions_test.csv"
    if y_test is not None:
        pd.DataFrame({"y_true": y_test, "y_pred": y_test_pred}).to_csv(test_pred_path, index=False)
    else:
        pd.DataFrame({"y_pred": y_test_pred}).to_csv(test_pred_path, index=False)

    logger.info("Saved evaluation files.")

    # ============================================================
    # MLFLOW LOGGING — MODEL + SUPPORTING ARTIFACTS
    # ============================================================
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"stage3_eval_{best_model_name}_{ts}"

    ARTIFACT_PATH = "model"   # folder name inside mlflow artifacts

    from mlflow import MlflowClient
    client = MlflowClient()

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Save run_information.json for register_model.py & app.py
        run_info = {
            "run_id": run_id,
            "model_name": f"HomePrice_{best_model_name}",   # stable model name
            "artifact_path": ARTIFACT_PATH
        }
        save_json(run_info, run_info_path)
        logger.info(f"Saved run information: {run_info_path}")


        logger.info(f"MLflow run started | run_id={run_id}")

        mlflow.set_tag("stage", "stage3+evaluation_swiggy_style")
        mlflow.log_param("target_col", TARGET)
        mlflow.log_param("best_model_name", best_model_name)

        mlflow.log_metrics({f"train_{k.lower()}": v for k, v in train_metrics.items()})
        if test_metrics:
            mlflow.log_metrics({f"test_{k.lower()}": v for k, v in test_metrics.items()})

        import mlflow.sklearn

        # LOG MODEL TO ARTIFACT PATH = "model"
        logger.info(f"Logging MLflow model at artifact_path='{ARTIFACT_PATH}' ...")

        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path=ARTIFACT_PATH,
            registered_model_name=None
        )

        # Verify artifact exists right after logging
        logger.info("Verifying that model artifact folder exists in MLflow...")
        # Verify artifacts by listing inside the model folder
        try:
            model_artifacts = client.list_artifacts(run_id, path=ARTIFACT_PATH)
            model_artifact_names = [a.path for a in model_artifacts]

            logger.info(f"Artifacts inside '{ARTIFACT_PATH}': {model_artifact_names}")

            if len(model_artifact_names) == 0:
                raise RuntimeError(
                    f"Model folder '{ARTIFACT_PATH}' exists but is empty. "
                    f"This indicates artifact upload failed."
                )

            logger.info("Model artifacts verified inside artifact folder.")

        except Exception as e:
            logger.warning(
                f"Could not verify model artifacts via list_artifacts due to: {repr(e)}. "
                "Skipping strict verification because DagsHub can return empty listing even when artifacts exist."
            )


        # Log other supporting artifacts
        mlflow.log_artifact(str(metrics_json_path))
        mlflow.log_artifact(str(train_pred_path))
        mlflow.log_artifact(str(test_pred_path))
        mlflow.log_artifact(str(preprocessor_path))
        mlflow.log_artifact(str(cleaner_path))
        mlflow.log_artifact(str(te_path))

        logger.info("MLflow logging complete")


