#custom_transformers
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer


class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        X_df = self._to_df(X)

        # ✅ if columns not provided, use input columns
        self.columns_ = self.columns if self.columns is not None else X_df.columns.tolist()

        self.mlbs_ = {}
        for col in self.columns_:
            mlb = MultiLabelBinarizer()
            values = X_df[col].fillna("").astype(str).str.split(", ")
            mlb.fit(values)
            self.mlbs_[col] = mlb

        self.fitted_ = True
        return self

    def _to_df(self, X):
        if isinstance(X, pd.DataFrame):
            return X

        # if ndarray -> create DF using fitted columns
        if not hasattr(self, "columns_"):
            raise ValueError("Got ndarray but columns_ not set. Call fit with DataFrame.")
        return pd.DataFrame(X, columns=self.columns_)

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, attributes=["fitted_"])

        X_df = self._to_df(X).copy()

        for col in self.columns_:
            mlb = self.mlbs_[col]
            split_vals = X_df[col].fillna("").astype(str).str.split(", ")
            arr = mlb.transform(split_vals)

            new_cols = [f"{col}__{c}" for c in mlb.classes_]
            arr_df = pd.DataFrame(arr, columns=new_cols, index=X_df.index)

            X_df = X_df.drop(columns=[col]).join(arr_df)

        return X_df




class FrequencyThresholdCategoriesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, thresholds=None, default_threshold=20, other_label="__other__"):
        self.thresholds = thresholds or {}
        self.default_threshold = default_threshold
        self.other_label = other_label

    def _to_df(self, X):
        # if already df, keep
        if isinstance(X, pd.DataFrame):
            return X

        # otherwise wrap into df using stored columns
        return pd.DataFrame(X, columns=self.columns_)

    def fit(self, X, y=None):
        # force dataframe
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns.tolist()
        else:
            # if fit is called with numpy, columns must already exist (rare)
            if not hasattr(self, "columns_"):
                raise ValueError("FrequencyThresholdCategoriesTransformer.fit got ndarray but columns_ not set.")

        X_df = self._to_df(X)
        #print("DEBUG type(X_df) =", type(X_df))


        self.valid_categories_ = {}
        for col in self.columns_:
            min_count = self.thresholds.get(col, self.default_threshold)
            counts = X_df[col].value_counts(dropna=True)
            self.valid_categories_[col] = set(counts[counts >= min_count].index)

        self.fitted_ = True
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, attributes=["fitted_"])

        X_df = self._to_df(X).copy()
        #print("DEBUG type(X) =", type(X))

        for col in self.columns_:
            valid = self.valid_categories_[col]
            X_df[col] = X_df[col].where(
                X_df[col].isna() | X_df[col].isin(valid),
                other=self.other_label
            )
        return X_df



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

    def _make_string(self, s: pd.Series) -> pd.Series:
        """
        Force safe string categories:
        - None -> 'missing'
        - NaN -> 'missing'
        - non-string -> str(...)
        """
        s = s.copy()
        s = s.replace({None: np.nan})
        s = s.astype("string")  # pandas string dtype
        s = s.fillna("missing").str.strip().str.lower()
        s = s.replace("", "missing")
        return s

    def fit(self, X, y=None):
        X = X.copy()
        X[self.column] = self._make_string(X[self.column])
        self.encoder.fit(X[[self.column]])
        self.fitted_ = True
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, attributes=["fitted_"])

        X = X.copy()
        X[self.column] = self._make_string(X[self.column])
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
    


# class ToDataFrame(BaseEstimator, TransformerMixin):
#     def __init__(self, columns=None):
#         self.columns = columns

#     def fit(self, X, y=None):
#         # save incoming column names
#         if hasattr(X, "columns"):
#             self.columns_ = list(X.columns)
#         else:
#             self.columns_ = self.columns
#         return self

#     def transform(self, X):
#         if isinstance(X, pd.DataFrame):
#             return X
#         return pd.DataFrame(X, columns=self.columns_)
