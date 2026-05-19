# ===============================
# analysis_agent.py
# ===============================

def price_analysis(row):
    try:
        min_p = row.get("buy_min", 0)
        max_p = row.get("buy_max", 0)
        actual = row.get("costpersqft", 0)

        if min_p == 0 or max_p == 0:
            return None

        mid = (min_p + max_p) / 2
        deviation = (actual - mid) / mid

        if actual > max_p:
            return {
                "flag": "overpriced",
                "severity": "high" if deviation > 0.15 else "medium",
                "message": f"Overpriced by {round(deviation*100,1)}%"
            }

        elif actual < min_p:
            return {
                "flag": "undervalued",
                "severity": "medium",
                "message": f"Underpriced by {round(abs(deviation)*100,1)}%"
            }

        else:
            return {
                "flag": "fair",
                "severity": "low",
                "message": "Within fair price range"
            }

    except:
        return None


def run_analysis(df):
    """
    Apply analysis to dataframe
    """
    results = []

    for _, row in df.iterrows():
        analysis = price_analysis(row)

        results.append({
            "id": row.get("id"),
            "analysis_flag": analysis["flag"] if analysis else None,
            "analysis_msg": analysis["message"] if analysis else None,
            "analysis_severity": analysis["severity"] if analysis else None
        })

    return results