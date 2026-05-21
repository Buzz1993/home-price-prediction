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
    Returns only matching properties from the dataframe based on filters.
    """
    
    temp = df.copy()

    for k, v in filters.items():
        if v == "Any" or v is None:
            continue

        if k in ["location", "transportation_hubs", "builder"]: # Not exact but Partial matching for text fields also allow - eg: for builder lodha as - Lodha Group,Lodha Builders,Lodha Crown
            temp = temp[temp[k].str.contains(str(v), case=False, na=False)]
        else:
            temp = temp[temp[k] == v] #Exact matching for normal columns

    return temp


# -----------------------------
# RECOMMEND
# -----------------------------
def recommend_with_constraints(df, X, filters, mode="static", k=10):
    """
    Generate similar property recommendations
    using filters + cosine similarity.

    Returns:
    - input property
    - top similar properties
    """

    temp = filter_data(df, filters) #using filters select the matching properties for that filters, from the main entire dataframe

    if len(temp) == 0:
        return None

    idx = np.random.choice(temp.index) # Pick one random property from matching properties
    vec = X[idx]  # get the vector for the chosen input property from the transformed matrix

    # if use _X means it contain numberic vectors dataframe and _df means dataframe containing real property details
    if mode == "static":
        compare_X = X[temp.index] # Take only matching property vectors from full feature matrix
        compare_df = temp
    else:
        compare_X = X # In dynamic mode, compare with all property vectors from full dataset
        compare_df = df

    sims = cosine_similarity(vec, compare_X).ravel()
    order = np.argsort(sims)[::-1] # Sort similarity scores from highest to lowest

    if mode == "static":
        order = order[1:] # Remove input property itself from recommendations
    else:
        order = order[order != idx]  # In dynamic mode, find and remove input property index

    top = order[:k]  # Select top 10 most similar property indexes

    sim_df = compare_df.iloc[top].copy() # Get real property details for top 10 recommended properties
    sim_df["cosine_similarity"] = sims[top] #add cosine similarity column in sim_df real property details dataframe with cosine similarity values

    inp = df.loc[[idx]].copy() # Get real property details for input selected property
    inp["cosine_similarity"] = 1.0 # Assign cosine similarity 1.0 always because property is compared with itself

    return {"input": inp, "similar": sim_df}