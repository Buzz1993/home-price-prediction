# ===============================
# content_based_filtering.py
# ===============================

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# PIPELINE
# -----------------------------
def build_pipeline():

    numeric_features = [
        'locality_rank','locality_rating',"environment_rating","commuting_rating",
        "places_of_interest_rating",'bath','parking','area','flat_on_floor',
        'total_floor','balcony','project_in_acres',"project_age_months",
        'education_min_km','education_mean_km','shopping_centre_mean_km',
        'shopping_centre_min_km','commercial_hub_mean_km','commercial_hub_min_km',
        'costpersqft','distance_to_center_km',
        'education_within_2km','transport_within_2km','shopping_centre_within_2km',
        'commercial_hub_within_2km','hospital_within_2km','tourist_within_2km',
        'total_within_2km','amenities_count','lift'
    ]

    categorical_ohe = ['property_type','ownership','facing','city','furnish','status']
    location_cols = ['builder','location']
    amenity_score_col = ['assigned_amenities_score']

    preprocessor = make_column_transformer(
        (Pipeline([('impute', SimpleImputer(strategy="median"))]), numeric_features),

        (Pipeline([
            ('impute', SimpleImputer(strategy="most_frequent")),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_ohe),

        (Pipeline([
            ('impute', SimpleImputer(strategy="constant", fill_value="missing")),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), location_cols),

        (Pipeline([
            ('impute', SimpleImputer(strategy="median"))
        ]), amenity_score_col),
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("normalize", Normalizer())
    ])


# -----------------------------
# TRAIN
# -----------------------------
def train(df):
    pipe = build_pipeline()
    X = pipe.fit_transform(df)
    return pipe, X


# -----------------------------
# FILTER
# -----------------------------
def filter_data(df, filters):
    """
    Filters the input DataFrame based on user-selected criteria.

    - Ignores filters with value "Any" or None
    - Uses partial matching (str.contains) for text fields like
      location, builder, and transportation_hubs
    - Uses exact matching (==) for all other fields

    Returns:
        Filtered DataFrame containing only matching rows
    """
    
    temp = df.copy()

    for k, v in filters.items():
        if v == "Any" or v is None:
            continue

        if k in ["location", "transportation_hubs", "builder"]:
            temp = temp[temp[k].str.contains(str(v), case=False, na=False)]
        else:
            temp = temp[temp[k] == v]

    return temp


# -----------------------------
# RECOMMEND
# -----------------------------
def recommend_with_constraints(df, X, filters, mode="static", k=10):

    temp = filter_data(df, filters)

    if len(temp) == 0:
        return None

    idx = np.random.choice(temp.index)
    vec = X[idx]

    if mode == "static":
        compare_X = X[temp.index]
        compare_df = temp
    else:
        compare_X = X
        compare_df = df

    sims = cosine_similarity(vec, compare_X).ravel()
    order = np.argsort(sims)[::-1]

    if mode == "static":
        order = order[1:]
    else:
        order = order[order != idx]

    top = order[:k]

    sim_df = compare_df.iloc[top].copy()
    sim_df["cosine_similarity"] = sims[top]

    inp = df.loc[[idx]].copy()
    inp["cosine_similarity"] = 1.0

    return {"input": inp, "similar": sim_df}