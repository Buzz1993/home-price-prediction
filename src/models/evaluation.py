# ============================================================
# evaluation.py
# ------------------------------------------------------------
# Stage 3 + Evaluation in ONE script:
#
# Stage 3:
# - Load Stage1 artifact best_trial_params_auto.pkl
# - Train FULL Target Encoding artifacts
# - Train FINAL pipeline on full train
# - Save:
#    - final_pipeline_best.pkl
#    - cleaner_full_for_te.pkl
#    - te_full.pkl
#
# Evaluation:
# - Load train/test
# - Apply TE
# - Predict using final_pipeline_best.pkl
# - Compute metrics Train & Test
# - Save metrics + predictions
# - Log all to MLflow (DagsHub)
# ============================================================

import os
import json
import yaml
import joblib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin, clone
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
# TRANSFORMERS (same as preprocess_train.py)
# ============================================================
class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        from sklearn.preprocessing import MultiLabelBinarizer
        self.mlb_ = {}
        self.columns_ = X.columns.tolist()

        for col in self.columns_:
            mlb = MultiLabelBinarizer()
            mlb.fit(X[col].fillna("").astype(str).str.split(", "))
            self.mlb_[col] = mlb

        self.fitted_ = True
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, attributes=["fitted_"])

        dfs = []
        for col, mlb in self.mlb_.items():
            arr = mlb.transform(X[col].fillna("").astype(str).str.split(", "))
            cols = [f"{col}_{c}" for c in mlb.classes_]
            dfs.append(pd.DataFrame(arr, columns=cols, index=X.index))

        return pd.concat(dfs, axis=1)


class FrequencyThresholdCategoriesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, thresholds=None, default_threshold=20, other_label="__other__"):
        self.thresholds = thresholds or {}
        self.default_threshold = default_threshold
        self.other_label = other_label

    def fit(self, X, y=None):
        self.columns_ = X.columns.tolist()
        self.valid_categories_ = {}
        for col in self.columns_:
            min_count = self.thresholds.get(col, self.default_threshold)
            counts = X[col].value_counts(dropna=True)
            self.valid_categories_[col] = set(counts[counts >= min_count].index)
        self.fitted_ = True
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, attributes=["fitted_"])

        X = X.copy()
        for col in self.columns_:
            valid = self.valid_categories_[col]
            X[col] = X[col].where(
                X[col].isna() | X[col].isin(valid),
                other=self.other_label
            )
        return X


class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, attributes=["fitted_"])

        X = X.copy()
        X[f"{self.column}_missing"] = X[self.column].isna().astype(int)
        return X


class ImputeAndScaleAmenity(BaseEstimator, TransformerMixin):
    def __init__(self, column="assigned_amenities_score", n_neighbors=5):
        self.column = column
        self.n_neighbors = n_neighbors
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.imputer.fit(X[[self.column]])
        imputed = self.imputer.transform(X[[self.column]])
        self.scaler.fit(imputed)
        self.fitted_ = True
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, attributes=["fitted_"])

        X = X.copy()
        imputed = self.imputer.transform(X[[self.column]])
        X[self.column] = self.scaler.transform(imputed)
        return X


class ConstructionOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column="construction", categories=None):
        self.column = column
        self.categories = categories
        self.encoder = OrdinalEncoder(
            categories=self.categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

    def fit(self, X, y=None):
        self.encoder.fit(X[[self.column]])
        self.fitted_ = True
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, attributes=["fitted_"])

        X = X.copy()
        X[self.column] = self.encoder.transform(X[[self.column]])
        return X


class CategoryCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        if self.cols is None:
            self.cols = [c for c, dt in X.dtypes.items() if dt == "object"]
        self.fitted_ = True
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, attributes=["fitted_"])

        X = X.copy()
        for c in self.cols:
            s = X[c].fillna("missing").astype(str).str.strip().str.lower()
            s = s.str.replace(r"[ \/\-]+", "_", regex=True)
            s = s.str.replace(r"[^0-9a-z_]", "", regex=True)
            s = s.str.replace(r"_+", "_", regex=True).str.strip("_")
            X[c] = s.replace("", "missing")
        return X


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
# PIPELINES (same as preprocess_train)
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

overlooking_extra_rooms_flooring_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy="constant", fill_value="missing")),
    ('multilable', MultiLabelBinarizerTransformer())
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
# Stage 3 Helpers
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

    # load data
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

    logger.info(f"Train loaded: {train_df.shape}")
    logger.info(f"Test loaded : {test_df.shape}")

    artifacts_dir = root_path / "cache" / "artifacts_auto"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = root_path / "cache" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # file paths
    stage1_best_params_path = artifacts_dir / "best_trial_params_auto.pkl"
    final_pipeline_path = artifacts_dir / "final_pipeline_best.pkl"
    cleaner_path = artifacts_dir / "cleaner_full_for_te.pkl"
    te_path = artifacts_dir / "te_full.pkl"

    if not stage1_best_params_path.exists():
        raise FileNotFoundError(f"Stage1 artifact not found: {stage1_best_params_path}")

    # ============================================================
    # STAGE 3: build final artifacts if missing
    # ============================================================
    if (not final_pipeline_path.exists()) or (not cleaner_path.exists()) or (not te_path.exists()):
        logger.info("Stage3 artifacts missing. Building FINAL model using Stage1 best params...")

        best_trial_params = joblib.load(stage1_best_params_path)
        best_model_name = best_trial_params["model"]
        best_params = {k: v for k, v in best_trial_params.items() if k != "model"}

        logger.info(f"Best model from Stage1: {best_model_name}")
        logger.info("Fitting TE artifacts on full train data...")

        ensure_cols_exist(X_train, te_cols, name="X_train")

        # Fit cleaner on full data
        cleaner_full_for_te = clone(builder_location_project_name_pipeline)
        cleaner_full_for_te.fit(X_train[te_cols])

        # Fit TE on full data (log target)
        y_train_log = np.log1p(y_train)
        X_train_clean = cleaner_full_for_te.transform(X_train[te_cols])
        te_full = TargetEncoder(smoothing=0.5)
        te_full.fit(X_train_clean, y_train_log)

        # Apply TE to X_train (Stage3)
        X_train_stage3 = apply_te_with_artifacts(X_train, cleaner_full_for_te, te_full, te_cols)

        # Fit final pipeline: preprocessor + model
        model = build_final_estimator(best_model_name, best_params, random_state=42)

        final_pipeline_best = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", model)
        ])

        logger.info("Training FINAL pipeline on full train...")
        final_pipeline_best.fit(X_train_stage3, y_train_log)

        # Save artifacts
        joblib.dump(final_pipeline_best, final_pipeline_path)
        joblib.dump(cleaner_full_for_te, cleaner_path)
        joblib.dump(te_full, te_path)

        logger.info("✅ Stage3 artifacts saved successfully.")
    else:
        logger.info("Stage3 artifacts already exist. Skipping build step.")

    # ============================================================
    # EVALUATION
    # ============================================================
    logger.info("Loading final artifacts...")
    final_pipeline = joblib.load(final_pipeline_path)
    cleaner_full = joblib.load(cleaner_path)
    te_full = joblib.load(te_path)

    ensure_cols_exist(X_train, te_cols, name="X_train")
    ensure_cols_exist(X_test, te_cols, name="X_test")

    logger.info("Applying Target Encoding on train & test...")
    X_train_te = apply_te_with_artifacts(X_train, cleaner_full, te_full, te_cols)
    X_test_te = apply_te_with_artifacts(X_test, cleaner_full, te_full, te_cols)

    logger.info("Generating predictions...")
    y_train_pred_log = final_pipeline.predict(X_train_te)
    y_train_pred = np.expm1(y_train_pred_log)

    train_metrics = calc_metrics(y_train, y_train_pred)
    logger.info(f"TRAIN metrics: {train_metrics}")

    test_metrics = None
    y_test_pred = None

    # Always save test preds even if y_test missing
    y_test_pred_log = final_pipeline.predict(X_test_te)
    y_test_pred = np.expm1(y_test_pred_log)

    if y_test is not None:
        test_metrics = calc_metrics(y_test, y_test_pred)
        logger.info(f"TEST metrics: {test_metrics}")
    else:
        logger.warning("No target column found in test.csv → skipping test metrics")

    # save metrics
    result = {
        "final_pipeline": final_pipeline_path.name,
        "target_col": TARGET,
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
    # MLFLOW LOGGING
    # ============================================================
    with mlflow.start_run() as run:
        mlflow.set_tag("stage", "stage3+evaluation")
        mlflow.set_tag("model_type", type(final_pipeline).__name__)
        mlflow.set_tag("target", TARGET)

        mlflow.log_param("target_col", TARGET)
        mlflow.log_param("te_cols", ",".join(te_cols))

        mlflow.log_metrics({f"train_{k.lower()}": v for k, v in train_metrics.items()})
        if test_metrics is not None:
            mlflow.log_metrics({f"test_{k.lower()}": v for k, v in test_metrics.items()})

        # log artifacts
        mlflow.log_artifact(str(metrics_json_path))
        mlflow.log_artifact(str(train_pred_path))
        mlflow.log_artifact(str(test_pred_path))

        mlflow.log_artifact(str(final_pipeline_path))
        mlflow.log_artifact(str(cleaner_path))
        mlflow.log_artifact(str(te_path))

        artifact_uri = mlflow.get_artifact_uri()
        run_id = run.info.run_id

        logger.info(f"MLflow logged Run ID: {run_id}")
        logger.info(f"Artifacts stored at: {artifact_uri}")

    # store run info
    run_info_path = root_path / "run_information.json"
    with open(run_info_path, "w") as f:
        json.dump(
            {"run_id": run_id, "artifact_uri": artifact_uri, "model_name": "final_pipeline_best"},
            f,
            indent=4
        )

    logger.info(f"Saved run info: {run_info_path}")
    logger.info("DONE Stage3 + Evaluation complete.")
