# ============================================================
# preprocess_train.py
# ------------------------------------------------------------
# SINGLE SCRIPT THAT DOES:
# 0) Load train/test from data/interim
# 1) Fold-wise Target Encoding (no leakage)
# 2) Fold-wise preprocessing fit/transform (no leakage)
# 3) STAGE 1: Optuna model selection (LGBM/XGB/RF/ET)
# 4) STAGE 2: Baseline CV Ensemble (OOF + test averaging)
#
# NOTE:
# - NO STAGE 3 included (as requested)
# - Reads config from params.yaml -> Train section
# ============================================================

import os
import joblib
import yaml
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import optuna

from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer

from category_encoders.target_encoder import TargetEncoder

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


# ============================================================
# GLOBAL CONFIG (defaults - overwritten by params.yaml)
# ============================================================
set_config(transform_output="pandas")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

TARGET = "price"
KNN_NEIGH = 5
N_TRIALS = 30
CV_FOLDS = 5
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 50

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)


# ============================================================
# LOGGER SETUP
# ============================================================
logger = logging.getLogger("preprocess_train")
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
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100


def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, mse, rmse, r2, mape


def read_params(file_path: Path) -> dict:
    with open(file_path, "r") as f:
        params_file = yaml.safe_load(f)
    return params_file


# ============================================================
# CUSTOM TRANSFORMERS
# ============================================================
class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        from sklearn.preprocessing import MultiLabelBinarizer
        self.mlb_ = {}
        self.columns_ = X.columns.tolist()

        for col in self.columns_:
            mlb = MultiLabelBinarizer()
            mlb.fit(X[col].fillna("").str.split(", "))
            self.mlb_[col] = mlb

        self.fitted_ = True
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, attributes=["fitted_"])

        dfs = []
        for col, mlb in self.mlb_.items():
            arr = mlb.transform(X[col].fillna("").str.split(", "))
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
        self.feature_names_in_ = X.columns.tolist()
        self.fitted_ = True
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, attributes=["fitted_"])

        X = X.copy()
        X[f"{self.column}_missing"] = X[self.column].isna().astype(int)
        return X


class ImputeAndScaleAmenity(BaseEstimator, TransformerMixin):
    def __init__(self, column='assigned_amenities_score', n_neighbors=5):

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
    def __init__(self, column='construction', categories=None):
        from sklearn.preprocessing import OrdinalEncoder

        self.column = column
        self.categories = categories
        self.encoder = OrdinalEncoder(
            categories=self.categories,
            handle_unknown='use_encoded_value',
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
# COLUMN GROUPS
# ============================================================
impute_topk_target_encoding_cols = ['builder', 'project_name', 'location']
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
    ("cat_clean", CategoryCleaner(cols=["builder", "project_name", "location"])),
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
    ('impute_and_scale', ImputeAndScaleAmenity(column='assigned_amenities_score', n_neighbors=KNN_NEIGH))
])

construction_pipeline = Pipeline([
    ('add_missing_indicator', MissingIndicatorAdder(column='construction')),
    ('ordinal_encode', ConstructionOrdinalEncoder(column='construction', categories=construction_categories))
])

city_seller_pipeline = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

distance_to_center_km_pipeline = Pipeline([
    ('impute', KNNImputer(n_neighbors=KNN_NEIGH)),
    ('scaler', MinMaxScaler())
])

within2km_pipeline = Pipeline([("scaler", StandardScaler())])

preprocessor = make_column_transformer(
    (KNNImputer(n_neighbors=KNN_NEIGH), features_to_fill_knn),
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
# TARGET ENCODING (FOLD SAFE)
# ============================================================
def add_fold_target_encoding(
    X_tr_raw: pd.DataFrame,
    y_tr_fold: np.ndarray,
    X_val_raw: pd.DataFrame,
    te_cols: list,
    cleaner_pipeline,
    X_test_raw: pd.DataFrame = None,
    smoothing: float = 0.5
):
    cleaner = clone(cleaner_pipeline)

    X_tr_clean = cleaner.fit_transform(X_tr_raw[te_cols])
    X_val_clean = cleaner.transform(X_val_raw[te_cols])
    X_test_clean = cleaner.transform(X_test_raw[te_cols]) if X_test_raw is not None else None

    X_tr = X_tr_raw.copy()
    X_val = X_val_raw.copy()
    X_test = X_test_raw.copy() if X_test_raw is not None else None

    y_tr_series = pd.Series(y_tr_fold, index=X_tr.index)
    global_mean = y_tr_series.mean()

    for col in te_cols:
        te = TargetEncoder(smoothing=smoothing)
        te.fit(X_tr_clean[[col]], y_tr_series)

        X_tr[f"{col}_te"] = te.transform(X_tr_clean[[col]]).iloc[:, 0].fillna(global_mean)
        X_val[f"{col}_te"] = te.transform(X_val_clean[[col]]).iloc[:, 0].fillna(global_mean)

        if X_test is not None:
            X_test[f"{col}_te"] = te.transform(X_test_clean[[col]]).iloc[:, 0].fillna(global_mean)

    X_tr.drop(columns=te_cols, inplace=True)
    X_val.drop(columns=te_cols, inplace=True)
    if X_test is not None:
        X_test.drop(columns=te_cols, inplace=True)

    return X_tr, X_val, X_test


# ============================================================
# OPTUNA MODEL BUILD
# ============================================================
def build_model_from_trial(trial, model_name):
    if model_name == "LGBM":
        return LGBMRegressor(
            n_estimators      = trial.suggest_int("n_estimators", 300, 1500),
            learning_rate     = trial.suggest_float("learning_rate", 0.006, 0.03),
            max_depth         = trial.suggest_int("max_depth", 4, 12),
            num_leaves        = trial.suggest_int("num_leaves", 16, 256),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 120),
            subsample         = trial.suggest_float("subsample", 0.5, 0.95),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 10.0),
            reg_lambda        = trial.suggest_float("reg_lambda", 0.0, 50.0),
            random_state      = RANDOM_STATE,
            n_jobs            = 1,
            verbosity         = -1,
        )
    elif model_name == "XGB":
        return XGBRegressor(
            n_estimators     = trial.suggest_int("n_estimators", 300, 1500),
            learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.05),
            max_depth        = trial.suggest_int("max_depth", 3, 12),
            subsample        = trial.suggest_float("subsample", 0.4, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 10.0),
            reg_lambda       = trial.suggest_float("reg_lambda", 0.0, 50.0),
            random_state     = RANDOM_STATE,
            n_jobs           = 1,
            verbosity        = 0,
            tree_method      = "hist",
        )
    elif model_name == "RF":
        return RandomForestRegressor(
            n_estimators      = trial.suggest_int("n_estimators", 200, 1000),
            max_depth         = trial.suggest_int("max_depth", 3, 30),
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 10),
            max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, None]),
            random_state      = RANDOM_STATE,
            n_jobs            = 1,
        )
    elif model_name == "ET":
        return ExtraTreesRegressor(
            n_estimators      = trial.suggest_int("n_estimators", 200, 1000),
            max_depth         = trial.suggest_int("max_depth", 3, 30),
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 10),
            max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, None]),
            random_state      = RANDOM_STATE,
            n_jobs            = 1,
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


def build_final_estimator(best_model_name, best_params):
    if best_model_name == "LGBM":
        return LGBMRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=1, verbosity=-1)
    elif best_model_name == "RF":
        return RandomForestRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=1)
    elif best_model_name == "ET":
        return ExtraTreesRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=1)
    elif best_model_name == "XGB":
        return XGBRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=1, verbosity=0, tree_method="hist")
    else:
        raise ValueError(f"Unknown model: {best_model_name}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    root_path = Path(__file__).parent.parent.parent

    # ---------------- params.yaml ----------------
    params_file_path = root_path / "params.yaml"
    params = read_params(params_file_path)

    # read Train section (your yaml)
    train_params = params.get("Train", {})

    TARGET = train_params.get("target_col", TARGET)
    n_trials = int(train_params.get("n_trials", N_TRIALS))
    cv_folds = int(train_params.get("cv_folds", CV_FOLDS))

    # overwrite globals
    RANDOM_STATE = int(train_params.get("random_state", RANDOM_STATE))
    EARLY_STOPPING_ROUNDS = int(train_params.get("early_stopping_rounds", EARLY_STOPPING_ROUNDS))
    KNN_NEIGH = int(train_params.get("knn_neighbors", KNN_NEIGH))

    np.random.seed(RANDOM_STATE)

    logger.info("Training parameters loaded from params.yaml")
    logger.info(f"TARGET={TARGET}, trials={n_trials}, folds={cv_folds}, "
                f"seed={RANDOM_STATE}, early_stop={EARLY_STOPPING_ROUNDS}, knn={KNN_NEIGH}")

    # ---------------- load train/test ----------------
    train_path = root_path / "data" / "interim" / "train.csv"
    test_path = root_path / "data" / "interim" / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if TARGET not in train_df.columns:
        raise ValueError(f"TARGET='{TARGET}' not found in train.csv")

    X_train = train_df.drop(columns=[TARGET]).copy()
    y_train = pd.Series(train_df[TARGET]).copy()

    X_test = test_df.drop(columns=[TARGET], errors="ignore").copy()
    y_test = pd.Series(test_df[TARGET]).copy() if TARGET in test_df.columns else None

    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test) if y_test is not None else None

    logger.info(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    # ============================================================
    # STAGE 1 — OPTUNA MODEL SELECTION
    # ============================================================
    print("\n" + "=" * 70)
    print("STAGE 1 — MODEL SELECTION (OPTUNA CV SEARCH)")
    print("=" * 70)

    cv_for_opt = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )

    pbar = tqdm(total=n_trials, desc=f"Optuna Auto Model Search ({n_trials} trials)", position=0, leave=True)
    best_per_family = {m: {"mae": np.inf, "params": None, "trial": None} for m in ["LGBM", "XGB", "RF", "ET"]}

    def objective(trial):
        model_name = trial.suggest_categorical("model", ["LGBM", "XGB", "RF", "ET"])
        model = build_model_from_trial(trial, model_name=model_name)

        fold_losses = []
        for train_idx, val_idx in cv_for_opt.split(X_train):
            X_tr_raw = X_train.iloc[train_idx].copy()
            X_val_raw = X_train.iloc[val_idx].copy()

            y_tr_fold = y_train_log.iloc[train_idx].values
            y_val_fold = y_train_log.iloc[val_idx].values

            X_tr_df, X_val_df, _ = add_fold_target_encoding(
                X_tr_raw, y_tr_fold, X_val_raw,
                te_cols=impute_topk_target_encoding_cols,
                cleaner_pipeline=builder_location_project_name_pipeline,
                X_test_raw=None
            )

            # ... inside the fold loop ...
            preproc = clone(preprocessor)

            # SEPARATE FIT AND TRANSFORM
            preproc.fit(X_tr_df, y_tr_fold) 
            X_tr_pre = preproc.transform(X_tr_df)
            X_val_pre = preproc.transform(X_val_df)


            if model_name in ("XGB", "LGBM"):
                fit_kwargs = dict(
                    eval_set=[(X_val_pre, y_val_fold)],
                    eval_metric="mae",
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose=False
                )
                try:
                    model.fit(X_tr_pre, y_tr_fold, **fit_kwargs)
                except TypeError:
                    model.fit(X_tr_pre, y_tr_fold)
            else:
                model.fit(X_tr_pre, y_tr_fold)

            y_val_pred_log = model.predict(X_val_pre)

            y_val_pred_orig = np.expm1(y_val_pred_log)
            y_val_true_orig = np.expm1(y_val_fold)

            loss = mean_absolute_error(y_val_true_orig, y_val_pred_orig)
            fold_losses.append(loss)

        avg_mae = float(np.mean(fold_losses))

        if avg_mae < best_per_family[model_name]["mae"]:
            best_per_family[model_name]["mae"] = avg_mae
            best_per_family[model_name]["params"] = trial.params.copy()
            best_per_family[model_name]["trial"] = trial.number

        return avg_mae

    def tqdm_callback(study_, trial_):
        pbar.update(1)
        if study_.best_value is not None:
            best_model = study_.best_trial.params.get("model", "")
            pbar.set_postfix({"Best MAE": f"{study_.best_value:.4f}", "Best Model": best_model})

    study.optimize(objective, n_trials=n_trials, n_jobs=1, callbacks=[tqdm_callback])
    pbar.close()

    best_trial = study.best_trial
    best_params = best_trial.params.copy()
    best_model_name = best_params.pop("model")

    print("\nOptuna search complete.")
    print(f"\n[STAGE 1 RESULT]")
    print(f"Best Trial      : {best_trial.number}")
    print(f"Best Model      : {best_model_name}")
    print(f"Best CV MAE     : {study.best_value:.4f}")
    print(f"Best Parameters : {best_params}")

    # Save artifacts
    artifacts_dir = os.path.join(CACHE_DIR, "artifacts_auto")
    os.makedirs(artifacts_dir, exist_ok=True)

    joblib.dump(best_trial.params, os.path.join(artifacts_dir, "best_trial_params_auto.pkl"))  
    joblib.dump(best_per_family, os.path.join(artifacts_dir, "best_per_family_summary_auto.pkl"))
    joblib.dump(study, os.path.join(artifacts_dir, "optuna_study_auto.pkl"))

    logger.info(f"Saved Optuna artifacts to: {artifacts_dir}")

    # ============================================================
    # STAGE 2 — BASELINE CV ENSEMBLE
    # ============================================================
    print("\n" + "=" * 70)
    print("STAGE 2 — BASELINE CV-ENSEMBLE (OOF + TEST AVERAGING)")
    print("=" * 70)

    print(f"\nTraining baseline CV-ensemble with {cv_folds} folds using: {best_model_name}")

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    oof_preds_baseline = np.zeros(X_train.shape[0], dtype=float)
    test_preds_accum_baseline = np.zeros(X_test.shape[0], dtype=float)

    fold = 0
    for train_idx, val_idx in kf.split(X_train):
        fold += 1
        print(f"\n-- Baseline Fold {fold}/{cv_folds} --")

        X_tr_raw = X_train.iloc[train_idx].copy()
        X_val_raw = X_train.iloc[val_idx].copy()

        y_tr_fold = y_train_log.iloc[train_idx].values
        y_val_fold = y_train_log.iloc[val_idx].values

        X_tr_df, X_val_df, X_test_df_te = add_fold_target_encoding(
            X_tr_raw, y_tr_fold, X_val_raw,
            te_cols=impute_topk_target_encoding_cols,
            cleaner_pipeline=builder_location_project_name_pipeline,
            X_test_raw=X_test.copy()
        )

        # ... inside Stage 2 fold loop ...
        preproc = clone(preprocessor)

        # SEPARATE FIT AND TRANSFORM
        preproc.fit(X_tr_df, y_tr_fold)
        X_tr_pre = preproc.transform(X_tr_df)
        X_val_pre = preproc.transform(X_val_df)
        X_test_pre = preproc.transform(X_test_df_te)


        model = build_final_estimator(best_model_name, best_params)

        if best_model_name in ("XGB", "LGBM"):
            fit_kwargs = dict(
                eval_set=[(X_val_pre, y_val_fold)],
                eval_metric="mae",
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=False
            )
            try:
                model.fit(X_tr_pre, y_tr_fold, **fit_kwargs)
            except TypeError:
                model.fit(X_tr_pre, y_tr_fold)
        else:
            model.fit(X_tr_pre, y_tr_fold)

        y_val_pred_log = model.predict(X_val_pre)
        test_pred_fold_log = model.predict(X_test_pre)

        oof_preds_baseline[val_idx] = y_val_pred_log
        test_preds_accum_baseline += test_pred_fold_log

    test_preds_avg_baseline = test_preds_accum_baseline / cv_folds

    # Train metrics
    y_pred_train_baseline = np.expm1(oof_preds_baseline)
    y_train_orig = np.expm1(y_train_log)
    train_metrics_baseline = calc_metrics(y_train_orig, y_pred_train_baseline)

    print("\n[STAGE 2 RESULT — BASELINE CV ENSEMBLE]")
    print("\n==== BASELINE CV-ENSEMBLE Train Metrics (OOF) ====")
    print(f"MAE: {train_metrics_baseline[0]:.4f} | MSE: {train_metrics_baseline[1]:.4f} "
          f"| RMSE: {train_metrics_baseline[2]:.4f} | R²: {train_metrics_baseline[3]:.4f} "
          f"| MAPE: {train_metrics_baseline[4]:.2f}%")

    if y_test is not None:
        y_pred_test_baseline = np.expm1(test_preds_avg_baseline)
        test_metrics_baseline = calc_metrics(y_test, y_pred_test_baseline)

        print("\n==== BASELINE CV-ENSEMBLE Test Metrics ====")
        print(f"MAE: {test_metrics_baseline[0]:.4f} | MSE: {test_metrics_baseline[1]:.4f} "
              f"| RMSE: {test_metrics_baseline[2]:.4f} | R²: {test_metrics_baseline[3]:.4f} "
              f"| MAPE: {test_metrics_baseline[4]:.2f}%")
    else:
        print("\nTest labels not found in test.csv, skipping test metrics.")

    # Save predictions
    pred_dir = os.path.join(CACHE_DIR, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    joblib.dump(oof_preds_baseline, os.path.join(pred_dir, "oof_preds_log_baseline.pkl"))
    joblib.dump(test_preds_avg_baseline, os.path.join(pred_dir, "test_preds_avg_log_baseline.pkl"))

    logger.info(f"Saved predictions to: {pred_dir}")

    print("\nDONE (Stage 1 + Stage 2 complete)")
