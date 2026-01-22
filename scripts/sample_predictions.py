# scripts/sample_predictions.py

import json
from pathlib import Path
import pandas as pd
import numpy as np
import requests

from target_utils import clean_price_target


def print_row_all_cols(df_row, title=""):
    print("\n" + "=" * 120)
    print(f"{title}")
    print("=" * 120)

    row = df_row.iloc[0]
    for col in df_row.columns:
        val = row[col]
        print(f"{col:<45} = {val}   (type={type(val)})")

    print("=" * 120 + "\n")


def main():
    root_path = Path(__file__).parent.parent
    data_path = root_path / "data" / "raw" / "f_original magicbricks cleaned 12022 data.csv"

    predict_url = "http://127.0.0.1:8000/predict"
    TARGET_COL = "PRICE"

    df = pd.read_csv(data_path, low_memory=False)
    df = clean_price_target(df, target_col=TARGET_COL)

    sample_row = df.sample(1)

    #print actual target always 
    target_val = float(sample_row[TARGET_COL].values.item())
    print(f"\nTarget value (cleaned, in Cr): {target_val}")

    input_row = sample_row.drop(columns=[TARGET_COL]).squeeze().to_dict()

    cleaned_input = {}
    for k, v in input_row.items():
        if pd.isna(v):
            continue
        if isinstance(v, (np.float32, np.float64)):
            v = float(v)
        if isinstance(v, (np.int32, np.int64)):
            v = int(v)
        cleaned_input[k] = v

    # call api
    response = requests.post(url=predict_url, json=cleaned_input, timeout=60)

    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2))

    else:
        # ERROR CASE: print full debug details
        print("\nERROR OCCURRED")
        print("Row index:", sample_row.index[0])
        print("Keys sent to API:", len(cleaned_input))

        # print all columns + values only in error
        print_row_all_cols(sample_row, "SAMPLED ROW (RAW)")

        # print request payload too (optional but useful)
        print("\n--- PAYLOAD SENT TO API ---")
        print(json.dumps(cleaned_input, indent=2, default=str))

        # print API response
        print("\n--- API ERROR RESPONSE ---")
        try:
            print(json.dumps(response.json(), indent=2))
        except Exception:
            print(response.text)


if __name__ == "__main__":
    main()
