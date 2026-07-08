# ===============================
# negotiation_agent.py
# ===============================

import pandas as pd


def safe_numeric(value, default=0):
    """
    Safely convert values like:
    "2.5%", "1,200", None, "", nan
    into numeric float values.
    """

    cleaned = (
        str(value)
        .replace("%", "")
        .replace(",", "")
        .strip()
    )

    numeric_value = pd.to_numeric(
        cleaned,
        errors="coerce"
    )

    if pd.isna(numeric_value):
        return default

    return numeric_value


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
    print("🟨 negotiation agent executed")

    results = []

    print("NEGOTIATION INPUT COLUMNS")
    print(df.columns.tolist())

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

        rental_yield = safe_numeric(row.get("rental_yield_percent", 0))

        demand_level = row.get("demand_level","")

        future_signals = row.get("future_signals","")

        locality_rank = row.get("locality_rank",0)

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

        if rental_yield < 2:
            power_score += 1

        if "low" in str(demand_level).lower():
            power_score += 2

        elif "medium" in str(demand_level).lower():
            power_score += 1

        if locality_rank > 50:
            power_score += 1

        # Prevent negative negotiation score
        power_score = max(0, power_score)

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
            if pd.notna(analysis_msg) and analysis_msg:
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

        if rental_yield < 2:
            strategy.append("Low rental yield weakens investment attractiveness")

        if "low" in str(demand_level).lower():
            strategy.append("Lower buyer demand may improve negotiation scope")

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
        if pd.notna(risk_categories) and risk_categories:
            talking_points.append(f"Detected risks: {risk_categories}")

        # add growth reason
        if pd.notna(growth_reason) and growth_reason:
            talking_points.append(growth_reason)

        # add analysis message
        if pd.notna(analysis_msg) and analysis_msg:
            talking_points.append(analysis_msg)

        talking_points.append(f"Demand level: {demand_level}")

        if pd.notna(future_signals) and future_signals:
            talking_points.append(f"Future infra signals: {future_signals}")

        # -----------------------------
        # FINAL OUTPUT
        # -----------------------------
        results.append({

            "id": row.get("id"),

            "negotiation_score": power_score,

            "negotiation_power": power,
            "suggested_discount_percent": f"{discount[0]}-{discount[1]}%",
            "target_price": round(target_price, 2),
            "price_position": price_position,
            "strategy": " | ".join(strategy),
            "talking_points": " | ".join(talking_points)
        })

    return pd.DataFrame(results)