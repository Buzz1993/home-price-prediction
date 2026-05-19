#negotiation_agent.py

import pandas as pd

def run_negotiation_agent(df: pd.DataFrame):
    """
    Analyze selected properties and generate
    negotiation insights such as:
    - negotiation power
    - expected discount range
    - target buying price
    - negotiation strategy
    - talking points

    Returns:
    - dataframe containing negotiation analysis
      for each property.
    """

    results = []

    for _, row in df.iterrows():

        price = row.get("price", 0)
        cost = row.get("costpersqft", 0)
        buy_min = row.get("buy_min", 0)
        buy_max = row.get("buy_max", 0)

        risk = row.get("risk_label", "")
        growth = row.get("growth_label", "")
        construction = str(row.get("construction", "")).lower()
        age = row.get("project_age_months", 0)

        locality = row.get("locality_rating", 0)
        amenities = row.get("amenities_count", 0)

        weaknesses = str(row.get("needs_improvement", "")).lower()

        # -----------------------------
        # 1. PRICE POSITION
        # -----------------------------
        if buy_max > 0 and price > buy_max:
            price_position = "overpriced"
        elif buy_min > 0 and price < buy_min:
            price_position = "underpriced"
        else:
            price_position = "fair"

        # -----------------------------
        # 2. NEGOTIATION POWER
        # -----------------------------
        power_score = 0

        if price_position == "overpriced":
            power_score += 2

        if "high risk" in risk.lower():
            power_score += 2
        elif "medium risk" in risk.lower():
            power_score += 1

        if "under construction" in construction:
            power_score += 1

        if age > 60:
            power_score += 1

        if weaknesses and weaknesses != "nan":
            power_score += 2

        if "high growth" in growth.lower():
            power_score -= 1

        if locality > 4:
            power_score -= 1

        # -----------------------------
        # 3. CLASSIFY POWER
        # -----------------------------
        if power_score >= 4:
            power = "High"
            discount = (8, 12)
        elif power_score >= 2:
            power = "Medium"
            discount = (5, 8)
        else:
            power = "Low"
            discount = (2, 5)

        # -----------------------------
        # 4. TARGET PRICE
        # -----------------------------
        avg_discount = sum(discount) / 2
        target_price = price * (1 - avg_discount / 100)

        # -----------------------------
        # 5. STRATEGY
        # -----------------------------
        strategy = []

        if price_position == "overpriced":
            strategy.append("Highlight that similar properties are priced lower")

        if "under construction" in construction:
            strategy.append("Use project delay risk as leverage")

        if weaknesses and weaknesses != "nan":
            strategy.append("Point out locality or infrastructure issues")

        if "high risk" in risk.lower():
            strategy.append("Mention risk factors to justify discount")

        if not strategy:
            strategy.append("Focus on closing quickly for a small discount")

        # -----------------------------
        # 6. TALKING POINTS
        # -----------------------------
        talking_points = [
            f"Comparable price range is {buy_min} - {buy_max}",
            f"Current price per sqft is {cost}",
            f"Project risk level is {risk}"
        ]

        results.append({
            "id": row["id"],
            "negotiation_power": power,
            "suggested_discount_percent": f"{discount[0]}-{discount[1]}%",
            "target_price": round(target_price, 2),
            "price_position": price_position,
            "strategy": " | ".join(strategy),
            "talking_points": " | ".join(talking_points)
        })

    return pd.DataFrame(results)