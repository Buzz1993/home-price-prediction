#rent_utils.py

def calculate_rent(row):
    """
    Calculates estimated minimum and maximum rent for a property.
    """
    try:
        rent_min = row.get("rent_min", 0)
        rent_max = row.get("rent_max", 0)

        area = row["area"]

        if rent_min == 0 or rent_max == 0 or area == 0:
            return None, None

        estimated_rent_min = int(rent_min * area)
        estimated_rent_max = int(rent_max * area)

        return estimated_rent_min, estimated_rent_max

    except:
        return None, None