#rent_utils.py

def calculate_rent(row):
    try:
        rent_min = row.get("rent_min", 0)
        rent_max = row.get("rent_max", 0)
        area = row.get("area", 0)

        if rent_min == 0 or rent_max == 0 or area == 0:
            return None, None

        min_rent = int(rent_min * area)
        max_rent = int(rent_max * area)

        return min_rent, max_rent

    except:
        return None, None