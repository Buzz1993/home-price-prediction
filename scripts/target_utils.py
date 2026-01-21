# scripts/target_utils.py
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("target_utils")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)


def clean_price_target(df: pd.DataFrame, target_col: str = "PRICE") -> pd.DataFrame:
    """
    Clean price target and return df with cleaned target.
    """
    df = df.copy()

    # --- Convert price to Cr ---
    def convert_price_to_cr(val):
        if isinstance(val, str):
            parts = val.replace("₹", "").strip().split()
            if len(parts) == 2:
                amount, unit = parts
                try:
                    amount = float(amount)
                except:
                    return np.nan
                return amount / 100 if unit.lower() == "lac" else amount
        return np.nan

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found. Available: {df.columns.tolist()}")

    df[target_col] = df[target_col].apply(convert_price_to_cr).astype("float64")

    # Drop missing prices
    df = df.dropna(subset=[target_col])

    return df
