#rental_agent.py

import pandas as pd

def run_rental_agent(df: pd.DataFrame):

    results = []

    for _, row in df.iterrows():

        price = row.get("price", 0)
        rent_min = row.get("rent_min", 0)
        rent_max = row.get("rent_max", 0)

        growth = str(row.get("growth_label", "")).lower()
        amenities = row.get("amenities_count", 0)
        transport = row.get("transport_within_2km", 0)
        commercial = row.get("commercial_hub_within_2km", 0)

        # -----------------------------
        # 1. RENT ESTIMATION
        # -----------------------------
        if rent_min > 0 and rent_max > 0:
            monthly_rent = ((rent_min + rent_max) / 2) * 1000
        else:
            # fallback (important)
            monthly_rent = (price * 10000000) * 0.0025 / 12
        annual_rent = monthly_rent * 12
    

        # -----------------------------
        # 2. RENTAL YIELD
        # -----------------------------
        if price > 0:
            rental_yield = (annual_rent / (price * 10000000)) * 100  # Cr → ₹
        else:
            rental_yield = 0

        # -----------------------------
        # 3. DEMAND SCORE
        # -----------------------------
        demand_score = 0

        if transport >= 1:
            demand_score += 1

        if commercial >= 1:
            demand_score += 1

        if amenities > 40:
            demand_score += 1

        if "high growth" in growth:
            demand_score += 1

        # classify demand
        if demand_score >= 3:
            demand = "High"
        elif demand_score == 2:
            demand = "Medium"
        else:
            demand = "Low"

        # -----------------------------
        # 4. INVESTMENT RATING
        # -----------------------------
        if rental_yield >= 3:
            rating = "Excellent"
        elif rental_yield >= 2:
            rating = "Good"
        elif rental_yield >= 1:
            rating = "Average"
        else:
            rating = "Low"

        # -----------------------------
        # 5. STRATEGY
        # -----------------------------
        if rating == "Good":
            strategy = "Strong rental investment opportunity"
        elif rating == "Average":
            strategy = "Consider for appreciation + rental mix"
        else:
            strategy = "Better for self-use than rental income"

        results.append({
            "id": row["id"],
            "monthly_rent_estimate": round(monthly_rent, 0),
            "annual_rent": round(annual_rent, 0),
            "rental_yield_percent": f"{rental_yield:.2f}%",
            "demand_level": demand,
            "investment_rating": rating,
            "rental_strategy": strategy
        })

    return pd.DataFrame(results)