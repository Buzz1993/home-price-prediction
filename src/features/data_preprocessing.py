import pandas as pd
import logging
from pathlib import Path
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    MinMaxScaler,
    StandardScaler,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn import set_config  

#-----------------Helper Utilities----------------

def log_df_info(df: pd.DataFrame, name: str):
    logger.info(
        f"{name} shape: rows={df.shape[0]}, cols={df.shape[1]}"
    )


def load_data(path: Path, name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.info(f"{name} loaded successfully from {path}")
        log_df_info(df, name)
        return df
    except FileNotFoundError:
        logger.error(f"{name} not found at {path}")
        raise


# =================================================
# sklearn config
# =================================================
set_config(transform_output="pandas")

# =================================================
# LOGGER
# =================================================
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# =================================================
# CONFIG
# =================================================
KNN_NEIGH = 5
TARGET_COL = "price"

# =================================================
# CUSTOM TRANSFORMERS
# =================================================

class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        from sklearn.preprocessing import MultiLabelBinarizer
        self.mlb_ = {}
        for col in X.columns:
            mlb = MultiLabelBinarizer()
            mlb.fit(X[col].fillna("").str.split(", "))
            self.mlb_[col] = mlb
        return self

    def transform(self, X):
        dfs = []
        for col, mlb in self.mlb_.items():
            arr = mlb.transform(X[col].fillna("").str.split(", "))
            cols = [f"{col}_{c}" for c in mlb.classes_]
            dfs.append(pd.DataFrame(arr, columns=cols, index=X.index))
        return pd.concat(dfs, axis=1)


class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[f"{self.column}_missing"] = X[self.column].isna().astype(int)
        return X


class AmenityImputerScaler(BaseEstimator, TransformerMixin):
    def __init__(self, column, n_neighbors=5):
        self.column = column
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors)
        self.scaler_ = StandardScaler()

        imputed = self.imputer_.fit_transform(X[[self.column]])
        self.scaler_.fit(imputed)
        return self

    def transform(self, X):
        X = X.copy()
        missing = X[self.column].isna().astype(int)

        imputed = self.imputer_.transform(X[[self.column]])
        scaled = self.scaler_.transform(imputed)

        return pd.DataFrame(
            {
                self.column: scaled.iloc[:, 0].values,
                f"{self.column}_missing": missing.values,
            },
            index=X.index,
        )



class ConstructionEncoderWithMissing(BaseEstimator, TransformerMixin):
    def __init__(self, categories):
        self.categories = categories
        self.encoder = OrdinalEncoder(
            categories=categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        missing = X.iloc[:, 0].isna().astype(int)
        encoded = self.encoder.transform(X)

        col = X.columns[0]
        return pd.DataFrame(
            {
                col: encoded.iloc[:, 0],
                f"{col}_missing": missing.values,
            },
            index=X.index,
        )




# =================================================
# COLUMN GROUPS
# =================================================

DROP_COLS = ["builder", "project_name", "location"]

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

features_to_fill_median = ["lift"]
features_to_fill_latlon = ["lattitude", "longitude"]

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
min_max_scaling = ['education_within_2km','transport_within_2km','shopping_centre_within_2km',
                   'commercial_hub_within_2km','hospital_within_2km','tourist_within_2km','total_within_2km']
knn_impute_scaling = ['distance_to_center_km']

# =================================================
# PREPROCESSOR
# =================================================
property_type_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy="most_frequent")),
    ('OHE', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

status_furnish_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy="most_frequent")),
    ('ordinal_encode', OrdinalEncoder(categories=ordinal_categories,
                                      handle_unknown='use_encoded_value',
                                      unknown_value=-1))
])

ownership_facing_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy="constant", fill_value="missing")),
    ('OHE', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

overlooking_extra_rooms_flooring_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy="constant", fill_value="missing")),
    ('multilabel', MultiLabelBinarizerTransformer())
])

assigned_amenities_pipeline = Pipeline([
    ('impute_and_scale', AmenityImputerScaler(column='assigned_amenities_score', n_neighbors=KNN_NEIGH))
])



city_seller_pipeline = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
distance_to_center_km_pipeline = Pipeline([
    ('impute', KNNImputer(n_neighbors=KNN_NEIGH)),
    ('scaler', MinMaxScaler())
])
within2km_pipeline = Pipeline([("scaler", StandardScaler())])

preprocessor = make_column_transformer(
    (KNNImputer(n_neighbors=KNN_NEIGH), features_to_fill_knn),
    (SimpleImputer(strategy="median"), features_to_fill_latlon),
    (SimpleImputer(strategy="median"), features_to_fill_median),
    (property_type_pipeline, impute_mf_and_OHE),
    (status_furnish_pipeline, impute_mf_and_ordinal_encode),
    (ownership_facing_pipeline, impute_missing_and_OHE),
    (overlooking_extra_rooms_flooring_pipeline, impute_missing_and_multilable),
    (assigned_amenities_pipeline, assignweight_missingindicator_KNNimputation_minmaxscale),
    (ConstructionEncoderWithMissing(construction_categories),['construction']),
    (city_seller_pipeline, onehotencode),
    (distance_to_center_km_pipeline, knn_impute_scaling),
    (within2km_pipeline, min_max_scaling),

    remainder="drop",
    verbose_feature_names_out=False,
    n_jobs=1,
)

# =================================================
# MAIN
# =================================================

if __name__ == "__main__":

    root = Path(__file__).parent.parent.parent

    # =================================================
    # LOAD DATA
    # =================================================
    train_path = root / "data/interim/train.csv"
    test_path  = root / "data/interim/test.csv"

    train_df = load_data(train_path, "Train dataset")
    test_df  = load_data(test_path, "Test dataset")

    # =================================================
    # SPLIT X & y
    # =================================================
    logger.info("Splitting features and target")

    X_train = train_df.drop(columns=[TARGET_COL] + DROP_COLS)
    y_train = train_df[TARGET_COL]

    X_test  = test_df.drop(columns=[TARGET_COL] + DROP_COLS)
    y_test  = test_df[TARGET_COL]

    log_df_info(X_train, "X_train")
    log_df_info(X_test, "X_test")

    # =================================================
    # FIT PREPROCESSOR
    # =================================================
    logger.info("Fitting preprocessor on training data")
    preprocessor.fit(X_train)

    logger.info("Preprocessor fitted successfully")

    # =================================================
    # TRANSFORM DATA
    # =================================================
    logger.info("Transforming training data")
    X_train_t = preprocessor.transform(X_train)
    log_df_info(X_train_t, "X_train_transformed")

    logger.info("Transforming test data")
    X_test_t = preprocessor.transform(X_test)
    log_df_info(X_test_t, "X_test_transformed")

    # =================================================
    # JOIN TARGET BACK
    # =================================================
    logger.info("Joining transformed features with target")

    train_out = X_train_t.join(y_train)
    test_out  = X_test_t.join(y_test)

    log_df_info(train_out, "Final train dataset")
    log_df_info(test_out, "Final test dataset")

    # =================================================
    # SAVE OUTPUTS
    # =================================================
    out_dir = root / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out_path = out_dir / "train_trans.csv"
    test_out_path  = out_dir / "test_trans.csv"

    train_out.to_csv(train_out_path, index=False)
    logger.info(f"Train transformed data saved to {train_out_path}")

    test_out.to_csv(test_out_path, index=False)
    logger.info(f"Test transformed data saved to {test_out_path}")

    # =================================================
    # SAVE PREPROCESSOR
    # =================================================
    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)

    preprocessor_path = model_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    logger.info(f"Preprocessor saved to {preprocessor_path}")
    logger.info("Data preprocessing pipeline completed successfully ✅")
