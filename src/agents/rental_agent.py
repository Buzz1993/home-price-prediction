# ===============================
# rental_agent.py
# ===============================

import pandas as pd


def run_rental_agent(df: pd.DataFrame):
    """
    Analyze rental potential of properties
    using:
    - rent estimation
    - rental yield
    - demand level
    - investment quality

    Returns:
        pd.DataFrame
    """

    print("🟨 rental agent executed")

    results = []

    for _, row in df.iterrows():

        # -----------------------------
        # PROPERTY DATA
        # -----------------------------
        price = row.get("price", 0)
        # -----------------------------
        # FUTURE AGENT OUTPUTS
        # -----------------------------
        growth_score = row.get("growth_score", 0)

        # -----------------------------
        # RISK AGENT OUTPUTS
        # -----------------------------
        risk_score = row.get("risk_score", 0)

        # -----------------------------
        # PROPERTY FEATURES
        # -----------------------------
        amenities = row.get("amenities_count", 0)

        transport = row.get("transport_within_2km", 0)

        commercial = row.get("commercial_hub_within_2km", 0)

        future_signals = row.get("future_signals","")

        locality_rank = row.get("locality_rank",0)

        distance_to_center = row.get("distance_to_center_km",0)

        locality_rating = row.get("locality_rating",0)

        # print("="*50)
        # print("dataframe data",df.columns.tolist())
        # print("="*50)


        # -----------------------------
        # 1. RENT ESTIMATION
        # -----------------------------

        estimated_rent_min = row.get("estimated_rent_min", 0)
        estimated_rent_max = row.get("estimated_rent_max", 0)

        # Use property-specific rent estimate
        if estimated_rent_min > 0 and estimated_rent_max > 0:
            monthly_rent = (
                estimated_rent_min +
                estimated_rent_max
            ) / 2

        else:
            # fallback rent estimation
            monthly_rent = (
                (price * 10000000) * 0.0025 / 12
            )

        annual_rent = monthly_rent * 12

        # -----------------------------
        # 2. RENTAL YIELD
        # -----------------------------
        if price > 0:
            # convert Cr → ₹
            rental_yield = (annual_rent/ (price * 10000000)) * 100

        else:
            rental_yield = 0

        # -----------------------------
        # 3. RENTAL DEMAND SCORE
        # -----------------------------
        demand_score = 0

        # good transport connectivity
        if transport >= 1:
            demand_score += 1

        # nearby commercial hubs
        if commercial >= 1:
            demand_score += 1

        # more amenities
        if amenities >= 15:
            demand_score += 1

        # future growth potential
        if growth_score >= 3:
            demand_score += 1

        # high risk reduces tenant demand
        if risk_score >= 6:
            demand_score -= 1

        if locality_rating >= 4:
            demand_score += 1

        if locality_rank > 50:
            demand_score -= 1

        if distance_to_center > 20:
            demand_score -= 1

        # Prevent negative demand score
        demand_score = max(0, demand_score)

        # -----------------------------
        # 4. DEMAND LEVEL
        # -----------------------------
        if demand_score >= 3:
            demand = "High"

        elif demand_score == 2:
            demand = "Medium"

        else:
            demand = "Low"

        # -----------------------------
        # 5. INVESTMENT RATING
        # -----------------------------
        if rental_yield >= 3:
            rating = "Excellent"

        elif rental_yield >= 2:
            rating = "Good"

        elif rental_yield >= 1:
            rating = "Average"

        else:
            rating = "Low"

        # strong growth improves rating
        if growth_score >= 3 and rating != "Excellent":

            if rating == "Low":
                rating = "Average"

            elif rating == "Average":
                rating = "Good"

        # high risk reduces rating
        if risk_score >= 6:

            if rating == "Excellent":
                rating = "Good"

            elif rating == "Good":
                rating = "Average"


        # -----------------------------
        # 6. RENTAL STRATEGY
        # -----------------------------
        if rating == "Excellent":
            strategy = ("Strong rental investment opportunity")

        elif rating == "Good":
            strategy = ("Balanced rental and appreciation opportunity")

        elif rating == "Average":
            strategy = ("Consider for appreciation + rental mix")

        else:
            strategy = ("Better for self-use than rental income")

        # high risk warning
        if risk_score >= 6:
            strategy += (" | High property risk may affect tenant demand")

        if pd.notna(future_signals) and future_signals:
            strategy += (
                f" | Future growth signals: "
                f"{future_signals}"
            )

        # -----------------------------
        # FINAL OUTPUT
        # -----------------------------
        results.append({

            "id": row.get("id"),
            "current_price": row.get("price"),

            "monthly_rent_estimate": round(monthly_rent, 0),
            "annual_rent": round(annual_rent, 0),

            "rental_yield_percent": f"{rental_yield:.2f}%",

            "demand_level": demand,
            "investment_rating": rating,

            "rental_strategy": strategy
        })

    return pd.DataFrame(results)