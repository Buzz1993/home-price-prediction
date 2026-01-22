#app.py 2
import os
import json
import joblib
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

import mlflow
import mlflow.sklearn
import dagshub
from sklearn.pipeline import Pipeline

from scripts.data_clean_utils import perform_property_data_cleaning

from sklearn import set_config
set_config(transform_output="pandas")



# =====================================================
# MLFLOW HTTP RETRY FIX (DagsHub 500 safe)
# =====================================================
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "30"
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "3"
os.environ["MLFLOW_HTTP_REQUEST_RETRY_DELAY"] = "1"


# =====================================================
# DAGSHUB + MLFLOW SETUP
# =====================================================
dagshub.init(
    repo_owner="bowlekarbhushan88",
    repo_name="home-price-prediction",
    mlflow=True
)

MLFLOW_URI = "https://dagshub.com/bowlekarbhushan88/home-price-prediction.mlflow"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_registry_uri(MLFLOW_URI)


# =====================================================
# PATHS
# =====================================================
ROOT_PATH = Path(__file__).parent
CACHE_DIR = ROOT_PATH / "cache"
ARTIFACTS_DIR = CACHE_DIR / "artifacts_auto"

RUN_INFO_PATH = CACHE_DIR / "run_information.json"
REGISTRY_INFO_PATH = CACHE_DIR / "model_registry" / "registered_model_info.json"

CLEANER_PATH = ARTIFACTS_DIR / "cleaner_full_for_te.pkl"
TE_PATH = ARTIFACTS_DIR / "te_full.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
LOCAL_MODEL_PATH = ARTIFACTS_DIR / "final_model.joblib"   # local backup

TE_COLS = ["builder", "project_name", "location"]



# =====================================================
# HELPERS
# =====================================================
def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def load_joblib(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing Joblib file: {path}")
    return joblib.load(path)


def print_df_debug(df: pd.DataFrame, title: str, max_rows: int = 1, max_cols: int = 60):
    """
    Print a dataframe snapshot (safe for 1 row).
    """
    print("\n" + "=" * 80)
    print(f"DEBUG DF: {title}")
    print("=" * 80)

    if df is None:
        print("DF is None")
        return

    print("Shape:", df.shape)

    if df.empty:
        print("DF is empty")
        return

    # show first rows transposed for readability
    show_df = df.head(max_rows)
    cols = list(show_df.columns)[:max_cols]
    show_df = show_df[cols]

    print(show_df.T)

    # Missingness
    na = df.isna().sum()
    top_na = na[na > 0].sort_values(ascending=False).head(20)
    if len(top_na) > 0:
        print("\nTop missing columns:")
        print(top_na)

    print("=" * 80)


def sanitize_for_json(obj):
    """
    Convert numpy types into python native so FastAPI can JSON serialize safely.
    """
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    return obj


# =====================================================
# LOAD LOCAL ARTIFACTS (TE + PREPROCESSOR)
# =====================================================
cleaner_full = load_joblib(CLEANER_PATH)
te_full = load_joblib(TE_PATH)
preprocessor = load_joblib(PREPROCESSOR_PATH)


def apply_te(X: pd.DataFrame) -> pd.DataFrame:
    """
    Apply target encoding using cleaner_full_for_te.pkl + te_full.pkl
    """
    X = X.copy()

    # Ensure TE_COLS exist (if not, create them as NaN)
    for col in TE_COLS:
        if col not in X.columns:
            X[col] = np.nan

    X_clean = cleaner_full.transform(X[TE_COLS].copy())
    X_te_df = te_full.transform(X_clean)

    for col in TE_COLS:
        X[f"{col}_te"] = X_te_df[col].values

    X.drop(columns=TE_COLS, inplace=True, errors="ignore")
    return X


# =====================================================
# LOAD MODEL INFO
# =====================================================
run_info = load_json(RUN_INFO_PATH)
model_name = run_info["model_name"]

# ✅ UPDATED: Use Alias instead of Stage
MODEL_ALIAS = "Production" 
# Format: models:/name@alias
registry_uri = f"models:/{model_name}@{MODEL_ALIAS}"


def load_model_safely():
    """
    1) Try Registry via Alias (@staging)
    2) Try Absolute Model Source (from cache)
    3) Local Fallback
    """
    print("Tracking URI:", mlflow.get_tracking_uri())

    # -------------------------------
    # 1) TRY REGISTRY
    # -------------------------------
    try:
        print(f"\n[1] Trying REGISTRY (Alias: @{MODEL_ALIAS}):", registry_uri)
        # MLflow 2.10+ automatically resolves @aliases in the URI
        model_ = mlflow.sklearn.load_model(registry_uri)
        print(f"✅ Loaded model from registry (Alias: @{MODEL_ALIAS})")
        return model_
    except Exception as e:
        print(f"Registry load failed (@{MODEL_ALIAS}):", repr(e))

    # -------------------------------
    # 2) TRY ABSOLUTE MODEL SOURCE
    # -------------------------------
    try:
        registry_info = load_json(REGISTRY_INFO_PATH)
        model_source = registry_info.get("model_source")
        if not model_source:
            raise ValueError("model_source missing in registered_model_info.json")

        print("\n[2] Trying MODEL SOURCE:", model_source)
        model_ = mlflow.sklearn.load_model(model_source)
        print("✅ Loaded model from model_source")
        return model_
    except Exception as e:
        print("model_source load failed:", repr(e))

    # -------------------------------
    # 3) LOCAL FALLBACK
    # -------------------------------
    try:
        print("\n[3] Loading LOCAL backup:", LOCAL_MODEL_PATH)
        model_ = joblib.load(LOCAL_MODEL_PATH)
        print("✅ Loaded local model backup")
        return model_
    except Exception as e:
        print("Local model load failed:", repr(e))
        raise RuntimeError("CRITICAL: All model loading attempts failed.")

model = load_model_safely()


# =====================================================
# BUILD PIPELINE: Preprocessor + Model
# =====================================================
model_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])


# =====================================================
# INPUT SCHEMA
# =====================================================
class Data(BaseModel):
    # Basic
    ID: Optional[str] = Field(default=None, alias="ID")
    at_type: Optional[str] = Field(default=None, alias="@type")
    at_id: Optional[str] = Field(default=None, alias="@id")
    url: Optional[str] = Field(default=None, alias="url")
    numberOfRooms: Optional[float] = Field(default=None, alias="numberOfRooms")
    image: Optional[Any] = Field(default=None, alias="image")
    name: Optional[str] = Field(default=None, alias="name")
    geo: Optional[Any] = Field(default=None, alias="geo")
    potentialAction: Optional[Any] = Field(default=None, alias="potentialAction")
    address: Optional[Any] = Field(default=None, alias="address")

    PRICE: Optional[str] = Field(default=None, alias="PRICE")
    BHK_Type: Optional[str] = Field(default=None, alias="BHK_Type")
    Area: Optional[str] = Field(default=None, alias="Area")
    property_loc: Optional[str] = Field(default=None, alias="property_loc")
    locality_URL: Optional[str] = Field(default=None, alias="locality_URL")
    EMI: Optional[str] = Field(default=None, alias="EMI")

    # MD_*
    MD_Price_Breakup: Optional[str] = Field(default=None, alias="MD_Price Breakup")
    MD_Booking_Amount: Optional[str] = Field(default=None, alias="MD_Booking Amount")
    MD_Address: Optional[str] = Field(default=None, alias="MD_Address")
    MD_Furnishing: Optional[str] = Field(default=None, alias="MD_Furnishing")
    MD_Flooring: Optional[str] = Field(default=None, alias="MD_Flooring")
    MD_Loan_Offered: Optional[str] = Field(default=None, alias="MD_Loan Offered")
    MD_Water_Availability: Optional[str] = Field(default=None, alias="MD_Water Availability")
    MD_RERA_ID: Optional[str] = Field(default=None, alias="MD_RERA ID")
    MD_Status_of_Electricity: Optional[str] = Field(default=None, alias="MD_Status of Electricity")
    MD_Lift: Optional[float] = Field(default=None, alias="MD_Lift")
    MD_Floors_allowed_for_construction: Optional[float] = Field(default=None, alias="MD_Floors allowed for construction")
    MD_Age_of_Construction: Optional[str] = Field(default=None, alias="MD_Age of Construction")
    MD_Landmarks: Optional[str] = Field(default=None, alias="MD_Landmarks")
    MD_Overlooking: Optional[str] = Field(default=None, alias="MD_Overlooking")
    MD_Type_of_Ownership: Optional[str] = Field(default=None, alias="MD_Type of Ownership")
    MD_Additional_Rooms: Optional[str] = Field(default=None, alias="MD_Additional Rooms")
    MD_Authority_Approval: Optional[str] = Field(default=None, alias="MD_Authority Approval")

    # AP_*
    AP_Price: Optional[str] = Field(default=None, alias="AP_Price")
    AP_Price_per_sqft: Optional[str] = Field(default=None, alias="AP_Price per sqft")
    AP_Configuration: Optional[str] = Field(default=None, alias="AP_Configuration")
    AP_Tower_Unit: Optional[str] = Field(default=None, alias="AP_Tower & Unit")
    AP_Pjt_URL: Optional[str] = Field(default=None, alias="AP_Pjt_URL")
    AP_Pjt_Name: Optional[str] = Field(default=None, alias="AP_Pjt_Name")
    AP_Buildr: Optional[str] = Field(default=None, alias="AP_Buildr")
    AP_Ratings: Optional[float] = Field(default=None, alias="AP_Ratings")
    AP_Reviews_by: Optional[str] = Field(default=None, alias="AP_Reviews_by")
    AP_Tower: Optional[str] = Field(default=None, alias="AP_Tower")
    AP_Unit: Optional[str] = Field(default=None, alias="AP_Unit")

    image_urls: Optional[str] = Field(default=None, alias="image_urls")
    headings_with_ratings: Optional[str] = Field(default=None, alias="headings_with_ratings")

    # About Project
    Aboutpjt_Project_Size: Optional[str] = Field(default=None, alias="Aboutpjt_Project Size")
    Aboutpjt_Launch_Date: Optional[str] = Field(default=None, alias="Aboutpjt_Launch Date")
    Aboutpjt_Total_Units: Optional[float] = Field(default=None, alias="Aboutpjt_Total Units")
    Aboutpjt_Total_Towers: Optional[float] = Field(default=None, alias="Aboutpjt_Total Towers")
    Aboutpjt_BHK: Optional[str] = Field(default=None, alias="Aboutpjt_BHK")
    Aboutpjt_Total_Floors: Optional[float] = Field(default=None, alias="Aboutpjt_Total Floors")

    # ---------------------------
    # AM_* columns
    # ---------------------------
    AM_12204: Optional[str] = Field(default=None, alias="AM_12204")
    AM_12226: Optional[str] = Field(default=None, alias="AM_12226")
    AM_12225: Optional[str] = Field(default=None, alias="AM_12225")
    AM_12229: Optional[str] = Field(default=None, alias="AM_12229")
    AM_12230: Optional[str] = Field(default=None, alias="AM_12230")
    AM_1404107: Optional[str] = Field(default=None, alias="AM_1404107")
    AM_12201: Optional[str] = Field(default=None, alias="AM_12201")
    AM_12205: Optional[str] = Field(default=None, alias="AM_12205")
    AM_12202: Optional[str] = Field(default=None, alias="AM_12202")
    AM_12209: Optional[str] = Field(default=None, alias="AM_12209")
    AM_12207: Optional[str] = Field(default=None, alias="AM_12207")
    AM_12208: Optional[str] = Field(default=None, alias="AM_12208")
    AM_12214: Optional[str] = Field(default=None, alias="AM_12214")
    AM_12216: Optional[str] = Field(default=None, alias="AM_12216")
    AM_12218: Optional[str] = Field(default=None, alias="AM_12218")
    AM_1404110: Optional[str] = Field(default=None, alias="AM_1404110")
    AM_12224: Optional[str] = Field(default=None, alias="AM_12224")
    AM_12206: Optional[str] = Field(default=None, alias="AM_12206")
    AM_1404117: Optional[str] = Field(default=None, alias="AM_1404117")
    AM_1404105: Optional[str] = Field(default=None, alias="AM_1404105")
    AM_12220: Optional[str] = Field(default=None, alias="AM_12220")
    AM_12228: Optional[str] = Field(default=None, alias="AM_12228")
    AM_12203: Optional[str] = Field(default=None, alias="AM_12203")
    AM_12211: Optional[str] = Field(default=None, alias="AM_12211")
    AM_1404118: Optional[str] = Field(default=None, alias="AM_1404118")
    AM_1404106: Optional[str] = Field(default=None, alias="AM_1404106")
    AM_1404125: Optional[str] = Field(default=None, alias="AM_1404125")
    AM_1404124: Optional[str] = Field(default=None, alias="AM_1404124")
    AM_12538: Optional[str] = Field(default=None, alias="AM_12538")
    AM_12540: Optional[str] = Field(default=None, alias="AM_12540")
    AM_12219: Optional[str] = Field(default=None, alias="AM_12219")
    AM_12227: Optional[str] = Field(default=None, alias="AM_12227")
    AM_12523: Optional[str] = Field(default=None, alias="AM_12523")
    AM_12533: Optional[str] = Field(default=None, alias="AM_12533")
    AM_12534: Optional[str] = Field(default=None, alias="AM_12534")
    AM_12535: Optional[str] = Field(default=None, alias="AM_12535")
    AM_12536: Optional[str] = Field(default=None, alias="AM_12536")
    AM_12537: Optional[str] = Field(default=None, alias="AM_12537")
    AM_12539: Optional[str] = Field(default=None, alias="AM_12539")
    AM_12543: Optional[str] = Field(default=None, alias="AM_12543")
    AM_12545: Optional[str] = Field(default=None, alias="AM_12545")
    AM_12581: Optional[str] = Field(default=None, alias="AM_12581")
    AM_12583: Optional[str] = Field(default=None, alias="AM_12583")
    AM_1404155: Optional[str] = Field(default=None, alias="AM_1404155")
    AM_1404114: Optional[str] = Field(default=None, alias="AM_1404114")
    AM_1404143: Optional[str] = Field(default=None, alias="AM_1404143")

    AM_1404109: Optional[str] = Field(default=None, alias="AM_1404109")
    AM_1404131: Optional[str] = Field(default=None, alias="AM_1404131")
    AM_12212: Optional[str] = Field(default=None, alias="AM_12212")
    AM_1404112: Optional[str] = Field(default=None, alias="AM_1404112")
    AM_12217: Optional[str] = Field(default=None, alias="AM_12217")
    AM_1404116: Optional[str] = Field(default=None, alias="AM_1404116")
    AM_1404120: Optional[str] = Field(default=None, alias="AM_1404120")
    AM_1404111: Optional[str] = Field(default=None, alias="AM_1404111")
    AM_12222: Optional[str] = Field(default=None, alias="AM_12222")
    AM_1404115: Optional[str] = Field(default=None, alias="AM_1404115")
    AM_1404123: Optional[str] = Field(default=None, alias="AM_1404123")
    AM_1404127: Optional[str] = Field(default=None, alias="AM_1404127")
    AM_1404128: Optional[str] = Field(default=None, alias="AM_1404128")
    AM_1404129: Optional[str] = Field(default=None, alias="AM_1404129")
    AM_12223: Optional[str] = Field(default=None, alias="AM_12223")
    AM_12215: Optional[str] = Field(default=None, alias="AM_12215")
    AM_1404108: Optional[str] = Field(default=None, alias="AM_1404108")
    AM_12213: Optional[str] = Field(default=None, alias="AM_12213")
    AM_12221: Optional[str] = Field(default=None, alias="AM_12221")
    AM_1404130: Optional[str] = Field(default=None, alias="AM_1404130")
    AM_1404126: Optional[str] = Field(default=None, alias="AM_1404126")
    AM_1404113: Optional[str] = Field(default=None, alias="AM_1404113")
    AM_1404122: Optional[str] = Field(default=None, alias="AM_1404122")

    AM_12586: Optional[str] = Field(default=None, alias="AM_12586")
    AM_1404156: Optional[str] = Field(default=None, alias="AM_1404156")
    AM_12525: Optional[str] = Field(default=None, alias="AM_12525")
    AM_12547: Optional[str] = Field(default=None, alias="AM_12547")
    AM_12577: Optional[str] = Field(default=None, alias="AM_12577")
    AM_1404150: Optional[str] = Field(default=None, alias="AM_1404150")
    AM_1404148: Optional[str] = Field(default=None, alias="AM_1404148")
    AM_1404146: Optional[str] = Field(default=None, alias="AM_1404146")

    AM_12529: Optional[str] = Field(default=None, alias="AM_12529")
    AM_12521: Optional[str] = Field(default=None, alias="AM_12521")
    AM_12522: Optional[str] = Field(default=None, alias="AM_12522")
    AM_12526: Optional[str] = Field(default=None, alias="AM_12526")
    AM_12530: Optional[str] = Field(default=None, alias="AM_12530")
    AM_12532: Optional[str] = Field(default=None, alias="AM_12532")
    AM_12528: Optional[str] = Field(default=None, alias="AM_12528")
    AM_12546: Optional[str] = Field(default=None, alias="AM_12546")
    AM_1404161: Optional[str] = Field(default=None, alias="AM_1404161")
    AM_12234: Optional[str] = Field(default=None, alias="AM_12234")
    AM_12585: Optional[str] = Field(default=None, alias="AM_12585")
    AM_1404149: Optional[str] = Field(default=None, alias="AM_1404149")
    AM_1404152: Optional[str] = Field(default=None, alias="AM_1404152")
    AM_1404158: Optional[str] = Field(default=None, alias="AM_1404158")
    AM_1404157: Optional[str] = Field(default=None, alias="AM_1404157")
    AM_1404154: Optional[str] = Field(default=None, alias="AM_1404154")
    AM_12579: Optional[str] = Field(default=None, alias="AM_12579")
    AM_12527: Optional[str] = Field(default=None, alias="AM_12527")
    AM_12524: Optional[str] = Field(default=None, alias="AM_12524")
    AM_12541: Optional[str] = Field(default=None, alias="AM_12541")
    AM_12238: Optional[str] = Field(default=None, alias="AM_12238")
    AM_12562: Optional[str] = Field(default=None, alias="AM_12562")
    AM_12578: Optional[str] = Field(default=None, alias="AM_12578")
    AM_12548: Optional[str] = Field(default=None, alias="AM_12548")
    AM_12549: Optional[str] = Field(default=None, alias="AM_12549")
    AM_12239: Optional[str] = Field(default=None, alias="AM_12239")
    AM_1404151: Optional[str] = Field(default=None, alias="AM_1404151")
    AM_12236: Optional[str] = Field(default=None, alias="AM_12236")
    AM_12237: Optional[str] = Field(default=None, alias="AM_12237")
    AM_12556: Optional[str] = Field(default=None, alias="AM_12556")
    AM_12560: Optional[str] = Field(default=None, alias="AM_12560")
    AM_12555: Optional[str] = Field(default=None, alias="AM_12555")
    AM_12561: Optional[str] = Field(default=None, alias="AM_12561")
    AM_1404147: Optional[str] = Field(default=None, alias="AM_1404147")
    AM_12235: Optional[str] = Field(default=None, alias="AM_12235")
    AM_12557: Optional[str] = Field(default=None, alias="AM_12557")
    AM_12233: Optional[str] = Field(default=None, alias="AM_12233")
    AM_12544: Optional[str] = Field(default=None, alias="AM_12544")
    AM_12232: Optional[str] = Field(default=None, alias="AM_12232")
    AM_12531: Optional[str] = Field(default=None, alias="AM_12531")
    AM_12518: Optional[str] = Field(default=None, alias="AM_12518")
    AM_12551: Optional[str] = Field(default=None, alias="AM_12551")
    AM_12554: Optional[str] = Field(default=None, alias="AM_12554")
    AM_12542: Optional[str] = Field(default=None, alias="AM_12542")
    AM_12520: Optional[str] = Field(default=None, alias="AM_12520")
    AM_12558: Optional[str] = Field(default=None, alias="AM_12558")
    AM_12584: Optional[str] = Field(default=None, alias="AM_12584")
    AM_1404160: Optional[str] = Field(default=None, alias="AM_1404160")
    AM_1404159: Optional[str] = Field(default=None, alias="AM_1404159")
    AM_12516: Optional[str] = Field(default=None, alias="AM_12516")
    AM_12580: Optional[str] = Field(default=None, alias="AM_12580")
    AM_12513: Optional[str] = Field(default=None, alias="AM_12513")
    AM_12550: Optional[str] = Field(default=None, alias="AM_12550")
    AM_12552: Optional[str] = Field(default=None, alias="AM_12552")
    AM_12553: Optional[str] = Field(default=None, alias="AM_12553")

    # ---------------------------
    # Places / hubs
    # ---------------------------
    Educational_Institute_1: Optional[str] = Field(default=None, alias="Educational Institute_1")
    Educational_Institute_2: Optional[str] = Field(default=None, alias="Educational Institute_2")
    Educational_Institute_3: Optional[str] = Field(default=None, alias="Educational Institute_3")
    Educational_Institute_4: Optional[str] = Field(default=None, alias="Educational Institute_4")
    Educational_Institute_5: Optional[str] = Field(default=None, alias="Educational Institute_5")

    Transportation_Hub_1: Optional[str] = Field(default=None, alias="Transportation Hub_1")
    Transportation_Hub_2: Optional[str] = Field(default=None, alias="Transportation Hub_2")
    Transportation_Hub_3: Optional[str] = Field(default=None, alias="Transportation Hub_3")
    Transportation_Hub_4: Optional[str] = Field(default=None, alias="Transportation Hub_4")
    Transportation_Hub_5: Optional[str] = Field(default=None, alias="Transportation Hub_5")

    Shopping_Centre_1: Optional[str] = Field(default=None, alias="Shopping Centre_1")
    Shopping_Centre_2: Optional[str] = Field(default=None, alias="Shopping Centre_2")
    Shopping_Centre_3: Optional[str] = Field(default=None, alias="Shopping Centre_3")
    Shopping_Centre_4: Optional[str] = Field(default=None, alias="Shopping Centre_4")
    Shopping_Centre_5: Optional[str] = Field(default=None, alias="Shopping Centre_5")

    Commercial_Hub_1: Optional[str] = Field(default=None, alias="Commercial Hub_1")
    Commercial_Hub_2: Optional[str] = Field(default=None, alias="Commercial Hub_2")
    Commercial_Hub_3: Optional[str] = Field(default=None, alias="Commercial Hub_3")
    Commercial_Hub_4: Optional[str] = Field(default=None, alias="Commercial Hub_4")
    Commercial_Hub_5: Optional[str] = Field(default=None, alias="Commercial Hub_5")

    Hospital_1: Optional[str] = Field(default=None, alias="Hospital_1")
    Hospital_2: Optional[str] = Field(default=None, alias="Hospital_2")
    Hospital_3: Optional[str] = Field(default=None, alias="Hospital_3")
    Hospital_4: Optional[str] = Field(default=None, alias="Hospital_4")
    Hospital_5: Optional[str] = Field(default=None, alias="Hospital_5")

    Tourist_Spot_1: Optional[str] = Field(default=None, alias="Tourist Spot_1")
    Tourist_Spot_2: Optional[str] = Field(default=None, alias="Tourist Spot_2")
    Tourist_Spot_3: Optional[str] = Field(default=None, alias="Tourist Spot_3")
    Tourist_Spot_4: Optional[str] = Field(default=None, alias="Tourist Spot_4")

    # locality
    locality_rank: Optional[float] = Field(default=None, alias="locality_rank")
    locality_URL_rating: Optional[float] = Field(default=None, alias="locality_URL_rating")
    locality_URL_review: Optional[float] = Field(default=None, alias="locality_URL_review")

    liv_Environment: Optional[str] = Field(default=None, alias="liv_Environment")
    liv_Commuting: Optional[str] = Field(default=None, alias="liv_Commuting")
    liv_Places_of_Interest: Optional[str] = Field(default=None, alias="liv_Places of Interest")

    # ---------------------------
    # BB_ columns (includes hyphen names)
    # ---------------------------
    BB_beds: Optional[float] = Field(default=None, alias="BB_beds")
    BB_baths: Optional[float] = Field(default=None, alias="BB_baths")
    BB_covered_parking: Optional[float] = Field(default=None, alias="BB_covered-parking")
    BB_unfurnished: Optional[float] = Field(default=None, alias="BB_unfurnished")

    BB_bed: Optional[float] = Field(default=None, alias="BB_bed")
    BB_bath: Optional[float] = Field(default=None, alias="BB_bath")

    BB_balcony: Optional[float] = Field(default=None, alias="BB_balcony")
    BB_balconies: Optional[float] = Field(default=None, alias="BB_balconies")
    BB_semi_furnished: Optional[float] = Field(default=None, alias="BB_semi-furnished")
    BB_furnished: Optional[float] = Field(default=None, alias="BB_furnished")

    # ---------------------------
    # many_ columns (spaces!)
    # ---------------------------
    many_Carpet_Area: Optional[str] = Field(default=None, alias="many_Carpet Area")
    many_Developer: Optional[str] = Field(default=None, alias="many_Developer")
    many_Project: Optional[str] = Field(default=None, alias="many_Project")
    many_Transaction_type: Optional[str] = Field(default=None, alias="many_Transaction type")
    many_Status: Optional[str] = Field(default=None, alias="many_Status")
    many_Lifts: Optional[float] = Field(default=None, alias="many_Lifts")
    many_Furnished_Status: Optional[str] = Field(default=None, alias="many_Furnished Status")
    many_Car_parking: Optional[str] = Field(default=None, alias="many_Car parking")
    many_Lift: Optional[float] = Field(default=None, alias="many_Lift")
    many_Floor: Optional[str] = Field(default=None, alias="many_Floor")
    many_Age_of_Construction: Optional[str] = Field(default=None, alias="many_Age of Construction")
    many_Super_Built_up_Area: Optional[str] = Field(default=None, alias="many_Super Built-up Area")
    many_Facing: Optional[str] = Field(default=None, alias="many_Facing")
    many_Type_of_Ownership: Optional[str] = Field(default=None, alias="many_Type of Ownership")
    many_Additional_Rooms: Optional[str] = Field(default=None, alias="many_Additional Rooms")
    many_Plot_Area: Optional[float] = Field(default=None, alias="many_Plot Area")

    # ---------------------------
    # leftBB_ columns (hyphen names)
    # ---------------------------
    leftBB_beds: Optional[float] = Field(default=None, alias="leftBB_beds")
    leftBB_baths: Optional[float] = Field(default=None, alias="leftBB_baths")
    leftBB_covered_parking: Optional[float] = Field(default=None, alias="leftBB_covered-parking")
    leftBB_unfurnished: Optional[float] = Field(default=None, alias="leftBB_unfurnished")

    leftBB_bed: Optional[float] = Field(default=None, alias="leftBB_bed")
    leftBB_bath: Optional[float] = Field(default=None, alias="leftBB_bath")

    leftBB_balcony: Optional[float] = Field(default=None, alias="leftBB_balcony")
    leftBB_balconies: Optional[float] = Field(default=None, alias="leftBB_balconies")
    leftBB_semi_furnished: Optional[float] = Field(default=None, alias="leftBB_semi-furnished")
    leftBB_furnished: Optional[float] = Field(default=None, alias="leftBB_furnished")

    # ---------------------------
    # leftmany_ columns (spaces!)
    # ---------------------------
    leftmany_Super_Built_up_Area: Optional[str] = Field(default=None, alias="leftmany_Super Built-up Area")
    leftmany_Developer: Optional[str] = Field(default=None, alias="leftmany_Developer")
    leftmany_Project: Optional[str] = Field(default=None, alias="leftmany_Project")
    leftmany_Transaction_type: Optional[str] = Field(default=None, alias="leftmany_Transaction type")
    leftmany_Status: Optional[str] = Field(default=None, alias="leftmany_Status")
    leftmany_Lifts: Optional[float] = Field(default=None, alias="leftmany_Lifts")
    leftmany_Furnished_Status: Optional[str] = Field(default=None, alias="leftmany_Furnished Status")
    leftmany_Car_parking: Optional[str] = Field(default=None, alias="leftmany_Car parking")
    leftmany_Carpet_Area: Optional[str] = Field(default=None, alias="leftmany_Carpet Area")
    leftmany_Floor: Optional[str] = Field(default=None, alias="leftmany_Floor")
    leftmany_Age_of_Construction: Optional[str] = Field(default=None, alias="leftmany_Age of Construction")
    leftmany_Facing: Optional[str] = Field(default=None, alias="leftmany_Facing")
    leftmany_Additional_Rooms: Optional[str] = Field(default=None, alias="leftmany_Additional Rooms")
    leftmany_Type_of_Ownership: Optional[str] = Field(default=None, alias="leftmany_Type of Ownership")
    leftmany_Lift: Optional[float] = Field(default=None, alias="leftmany_Lift")

    # ---------------------------
    # property type one-hot columns (starts with numbers / spaces)
    # ---------------------------
    type_2_BHK_Flat: Optional[str] = Field(default=None, alias="2 BHK Flat")
    type_3_BHK_Flat: Optional[str] = Field(default=None, alias="3 BHK Flat")
    type_1_BHK_Flat: Optional[str] = Field(default=None, alias="1 BHK Flat")
    type_4_BHK_Flat: Optional[str] = Field(default=None, alias="4 BHK Flat")
    type_5_BHK_Flat: Optional[str] = Field(default=None, alias="5 BHK Flat")
    type_6_BHK_Flat: Optional[str] = Field(default=None, alias="6 BHK Flat")

    type_Studio_Apartment: Optional[str] = Field(default=None, alias="Studio Apartment")
    type_Multistorey_Apartment: Optional[str] = Field(default=None, alias="Multistorey Apartment")
    type_Residential_Plot: Optional[str] = Field(default=None, alias="Residential Plot")
    type_Commercial_Office_Space: Optional[str] = Field(default=None, alias="Commercial Office Space")

    type_3_BHK_Villa: Optional[str] = Field(default=None, alias="3 BHK Villa")
    type_4_BHK_Villa: Optional[str] = Field(default=None, alias="4 BHK Villa")
    type_5_BHK_Villa: Optional[float] = Field(default=None, alias="5 BHK Villa")

    type_2_BHK_Builder: Optional[str] = Field(default=None, alias="2 BHK Builder")
    type_3_BHK_Builder: Optional[str] = Field(default=None, alias="3 BHK Builder")

    type_4_BHK_Penthouse: Optional[str] = Field(default=None, alias="4 BHK Penthouse")
    type_5_BHK_Penthouse: Optional[str] = Field(default=None, alias="5 BHK Penthouse")
    type_3_BHK_Penthouse: Optional[str] = Field(default=None, alias="3 BHK Penthouse")

    Rent: Optional[str] = Field(default=None, alias="Rent")

    class Config:
        populate_by_name = True
        extra = "allow"


# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI()


@app.get("/")
def home():
    return "Welcome to the Home Price Prediction API"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: Data):
    try:
        # ---------------------------------------------------------
        # 1) RAW INPUT DF
        # ---------------------------------------------------------
        X_raw = pd.DataFrame([data.model_dump(by_alias=True)])
        print_df_debug(X_raw, "RAW INPUT (FROM REQUEST)")

        # ---------------------------------------------------------
        # 2) CLEAN RAW -> MODEL FEATURES
        # ---------------------------------------------------------
        X_clean = perform_property_data_cleaning(X_raw)

        # ADD THIS LINE HERE:
        X_clean = X_clean.copy()
        
        print_df_debug(X_clean, "AFTER perform_property_data_cleaning()")

        if X_clean is None or X_clean.empty:
            raise ValueError("Cleaning returned empty dataframe.")

        # ---------------------------------------------------------
        # 3) APPLY TARGET ENCODING
        # ---------------------------------------------------------
        X_te = apply_te(X_clean)
        print_df_debug(X_te, "AFTER TARGET ENCODING (TE)")

        # ---------------------------------------------------------
        # 4) FINAL SAFETY CHECKS (None comparisons)
        # ---------------------------------------------------------
        # Convert object dtypes which look numeric safely:
        for col in X_te.columns:
            if X_te[col].dtype == "object":
                # Try coercing numeric-looking objects
                coerced = pd.to_numeric(X_te[col], errors="ignore")
                X_te[col] = coerced

        # Make sure no python None leaks in numeric cols
        X_te = X_te.replace({None: np.nan})

        # ---------------------------------------------------------
        # 5) PREDICT LOG PRICE
        # ---------------------------------------------------------
        pred_log = model_pipe.predict(X_te)[0]
        pred_log = float(pred_log)
        pred_price = float(np.expm1(pred_log))

        return {
            "predicted_price": round(pred_price, 2),
            "model_metadata": {
                "name": model_name,
                "alias": MODEL_ALIAS,  # Changed from stage
                "unit": "Cr"
            }
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()

        # Show full traceback in response for debugging
        raise HTTPException(
            status_code=400,
            detail={
                "error": str(e),
                "traceback": tb
            }
        )


if __name__ == "__main__":
    uvicorn.run(app="app:app", host="0.0.0.0", port=8000)
