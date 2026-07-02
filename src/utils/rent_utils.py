# ===============================
# rent_utils.py
# ===============================

import pandas as pd

def calculate_rent(row):
    """
    Calculates estimated minimum and maximum rent for a property.
    """
    try:
        #data validation and NaN handling before the rent calculation
        rent_min = pd.to_numeric(
            row.get("rent_min", 0),
            errors="coerce"
        )

        rent_max = pd.to_numeric(
            row.get("rent_max", 0),
            errors="coerce"
        )

        area = pd.to_numeric(
            row.get("area", 0),
            errors="coerce"
        )

        if (
            pd.isna(rent_min)
            or pd.isna(rent_max)
            or pd.isna(area)
            or rent_min <= 0
            or rent_max <= 0
            or area <= 0
        ):
            return 0, 0

        estimated_rent_min = int(rent_min * area)
        estimated_rent_max = int(rent_max * area)

        return estimated_rent_min, estimated_rent_max

    except Exception as e:
        print(
            "RENT ERROR:",
            row.get("id"),
            e
        )
        return 0, 0