# #negotiation_agent.py

# import pandas as pd

# def run_negotiation_agent(df: pd.DataFrame):
#     """
#     Analyze selected properties and generate
#     negotiation insights such as:
#     - negotiation power
#     - expected discount range
#     - target buying price
#     - negotiation strategy
#     - talking points

#     Returns:
#     - dataframe containing negotiation analysis
#       for each property.
#     """

#     results = []

#     for _, row in df.iterrows():

#         price = row.get("price", 0)
#         cost = row.get("costpersqft", 0)
#         buy_min = row.get("buy_min", 0)
#         buy_max = row.get("buy_max", 0)

#         risk = row.get("risk_label", "")
#         growth = row.get("growth_label", "")
#         construction = str(row.get("construction", "")).lower()
#         age = row.get("project_age_months", 0)

#         locality = row.get("locality_rating", 0)
#         amenities = row.get("amenities_count", 0)

#         weaknesses = str(row.get("needs_improvement", "")).lower()

#         # -----------------------------
#         # 1. PRICE POSITION
#         # -----------------------------
#         if buy_max > 0 and price > buy_max:
#             price_position = "overpriced"
#         elif buy_min > 0 and price < buy_min:
#             price_position = "underpriced"
#         else:
#             price_position = "fair"

#         # -----------------------------
#         # 2. NEGOTIATION POWER
#         # -----------------------------
#         power_score = 0

#         if price_position == "overpriced":
#             power_score += 2

#         if "high risk" in risk.lower():
#             power_score += 2
#         elif "medium risk" in risk.lower():
#             power_score += 1

#         if "under construction" in construction:
#             power_score += 1

#         if age > 60:
#             power_score += 1

#         if weaknesses and weaknesses != "nan":
#             power_score += 2

#         if "high growth" in growth.lower():
#             power_score -= 1

#         if locality > 4:
#             power_score -= 1

#         # -----------------------------
#         # 3. CLASSIFY POWER
#         # -----------------------------
#         if power_score >= 4:
#             power = "High"
#             discount = (8, 12)
#         elif power_score >= 2:
#             power = "Medium"
#             discount = (5, 8)
#         else:
#             power = "Low"
#             discount = (2, 5)

#         # -----------------------------
#         # 4. TARGET PRICE
#         # -----------------------------
#         avg_discount = sum(discount) / 2
#         target_price = price * (1 - avg_discount / 100)

#         # -----------------------------
#         # 5. STRATEGY
#         # -----------------------------
#         strategy = []

#         if price_position == "overpriced":
#             strategy.append("Highlight that similar properties are priced lower")

#         if "under construction" in construction:
#             strategy.append("Use project delay risk as leverage")

#         if weaknesses and weaknesses != "nan":
#             strategy.append("Point out locality or infrastructure issues")

#         if "high risk" in risk.lower():
#             strategy.append("Mention risk factors to justify discount")

#         if not strategy:
#             strategy.append("Focus on closing quickly for a small discount")

#         # -----------------------------
#         # 6. TALKING POINTS
#         # -----------------------------
#         talking_points = [
#             f"Comparable price range is {buy_min} - {buy_max}",
#             f"Current price per sqft is {cost}",
#             f"Project risk level is {risk}"
#         ]

#         results.append({
#             "id": row["id"],
#             "negotiation_power": power,
#             "suggested_discount_percent": f"{discount[0]}-{discount[1]}%",
#             "target_price": round(target_price, 2),
#             "price_position": price_position,
#             "strategy": " | ".join(strategy),
#             "talking_points": " | ".join(talking_points)
#         })

#     return pd.DataFrame(results)

#====================================================================================================================================================================================================================

# ===============================
# negotiation_agent.py
# ===============================

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
        pd.DataFrame
    """

    results = []

    for _, row in df.iterrows():

        # -----------------------------
        # PROPERTY DATA
        # -----------------------------
        price = row.get("price", 0)
        cost = row.get("costpersqft", 0)

        # buy_min = minimum buying price per sqft
        # buy_max = maximum buying price per sqft
        buy_min = row.get("buy_min", 0)
        buy_max = row.get("buy_max", 0)

        # -----------------------------
        # RISK AGENT OUTPUTS
        # -----------------------------
        risk = row.get("risk_label", "")

        risk_score = row.get("risk_score", 0)

        risk_categories = row.get("risk_categories", "")

        # -----------------------------
        # FUTURE AGENT OUTPUTS
        # -----------------------------
        growth = row.get("growth_label", "")

        growth_reason = row.get("growth_reason", "")

        growth_score = row.get("growth_score", 0)

        # -----------------------------
        # ANALYSIS AGENT OUTPUTS
        # -----------------------------
        analysis_flag = row.get("analysis_flag", "fair")

        analysis_msg = row.get("analysis_msg", "")

        analysis_severity = row.get("analysis_severity", "low")

        # -----------------------------
        # PROPERTY DETAILS
        # -----------------------------
        construction = str(row.get("construction", "")).lower()

        age = row.get("project_age_months", 0)

        locality = row.get("locality_rating", 0)

        amenities = row.get("amenities_count", 0)

        weaknesses = str(row.get("needs_improvement", "")).lower()

        # -----------------------------
        # 1. PRICE POSITION
        # -----------------------------
        # Use analysis agent output
        price_position = (analysis_flag or "fair")

        # -----------------------------
        # 2. NEGOTIATION POWER
        # -----------------------------
        power_score = 0

        # overpriced properties
        # give stronger negotiation power
        if analysis_flag == "overpriced":

            if analysis_severity == "high":
                power_score += 3

            else:
                power_score += 2

        # risk score impact
        if risk_score >= 6:
            power_score += 2

        elif risk_score >= 3:
            power_score += 1

        # under construction projects
        if "under construction" in construction:
            power_score += 1

        # older projects
        if age > 60:
            power_score += 1

        # locality/property issues
        if weaknesses and weaknesses != "nan":
            power_score += 2

        # high growth areas reduce leverage
        if growth_score >= 3:
            power_score -= 2

        elif growth_score >= 1:
            power_score -= 1

        # premium locality reduces leverage
        if locality > 4:
            power_score -= 1

        # more amenities reduce leverage
        if amenities > 15:
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
        avg_discount = (sum(discount) / 2)

        target_price = price * (1 - avg_discount / 100)

        # -----------------------------
        # 5. NEGOTIATION STRATEGY
        # -----------------------------
        strategy = []

        # overpriced property
        if analysis_flag == "overpriced":
            strategy.append("Use nearby comparable pricing as leverage")
            if analysis_msg:
                strategy.append(analysis_msg)

        # construction issues
        if "under construction" in construction:
            strategy.append("Use project delay risk as leverage")

        # locality/infrastructure issues
        if weaknesses and weaknesses != "nan":
            strategy.append("Use locality concerns during negotiation")

        # high risk property
        if risk_score >= 6:
            strategy.append("Use multiple risk factors as leverage")

        # high growth areas
        if growth_score >= 3:
            strategy.append("Future infrastructure growth may reduce seller flexibility")

        # fallback strategy
        if not strategy:
            strategy.append("Negotiate for a modest discount")

        # -----------------------------
        # 6. TALKING POINTS
        # -----------------------------
        talking_points = [

            f"Fair buying range: ₹{buy_min} - ₹{buy_max}/sqft",

            f"Current price: ₹{cost}/sqft",

            f"Risk level: {risk}",

            f"Growth outlook: {growth}"
        ]

        # add risk categories
        if risk_categories:
            talking_points.append(f"Detected risks: {risk_categories}")

        # add growth reason
        if growth_reason:
            talking_points.append(growth_reason)

        # add analysis message
        if analysis_msg:
            talking_points.append(analysis_msg)

        # -----------------------------
        # FINAL OUTPUT
        # -----------------------------
        results.append({

            "id": row.get("id"),
            "negotiation_power":power,
            "suggested_discount_percent":f"{discount[0]}-{discount[1]}%",
            "target_price":round(target_price, 2),
            "price_position":price_position,
            "strategy":" | ".join(strategy),
            "talking_points":" | ".join(talking_points)
        })

    return pd.DataFrame(results)