# ===============================
# analysis_agent.py
# ===============================

import pandas as pd


def price_analysis(row):
    """
    Analyze whether a property's cost per sqft
    is overpriced, undervalued, or fair
    based on buy_min and buy_max range.
    """
    try:

        # buy_min = minimum buying price per sqft
        # buy_max = maximum buying price per sqft
        min_p = row.get("buy_min", 0) #buy_min and buy_min already present in dataframe created while data cleaning
        max_p = row.get("buy_max", 0)
        actual = row.get("costpersqft", 0)

        print(
            f"ID={row.get('id')} | "
            f"buy_min={min_p} | "
            f"buy_max={max_p} | "
            f"costpersqft={actual}"
        )

        # Handle missing benchmark data
        if pd.isna(min_p) or pd.isna(max_p) or min_p <= 0 or max_p <= 0:
            print(f"⚠️ Missing benchmark for {row.get('id')}")
            return None

        mid = (min_p + max_p) / 2
        deviation = (actual - mid) / mid

        if actual > max_p:
            return {
                "flag": "overpriced",
                "severity": "high" if deviation > 0.15 else "medium",
                "message": f"Overpriced by {round(deviation * 100, 1)}%"
            }

        elif actual < min_p:
            return {
                "flag": "undervalued",
                "severity": "medium",
                "message": f"Underpriced by {round(abs(deviation) * 100, 1)}%"
            }

        else:
            return {
                "flag": "fair",
                "severity": "low",
                "message": "Within fair price range"
            }

    except Exception as e:
        print(f"❌ Price analysis error for {row.get('id')}: {e}")
        return None


def run_analysis(df):
    """
    Run price analysis for all properties
    and return analysis results as a list.
    """

    print("✅ analysis_agent executed")

    missing_benchmark_count = (
        df["buy_min"].isna()
        | df["buy_max"].isna()
        | (df["buy_min"] <= 0)
        | (df["buy_max"] <= 0)
    ).sum()

    print(f"Missing benchmark count: {missing_benchmark_count}")

    results = []

    for _, row in df.iterrows():
        analysis = price_analysis(row)

        results.append({
            "id": row.get("id"),
            "analysis_flag": analysis["flag"] if analysis else None, # create new column "analysis_flag" in results with value as analysis["flag"] (overpriced, undervalued, fair) if analysis exists else None
            "analysis_msg": analysis["message"] if analysis else None, # create new column "analysis_msg" in results with value as analysis["message"] (overpriced by x%, underpriced by x%, within fair price range) if analysis exists else None
            "analysis_severity": analysis["severity"] if analysis else None # create new column "analysis_severity" in results with value as analysis["severity"] (high, medium, low) if analysis exists else None
        })

    return results

    #ruturn as list
    #eg:[{'id': 'cardid71276787', 'analysis_flag': 'fair', 'analysis_msg': 'Within fair price range', 'analysis_severity': 'low'}, 
    # {'id': 'cardid71100181', 'analysis_flag': 'overpriced', 'analysis_msg': 'Overpriced by 50.5%', 'analysis_severity': 'high'}, 
    # {'id': 'cardid70034733', 'analysis_flag': 'fair', 'analysis_msg': 'Within fair price range', 'analysis_severity': 'low'},.... 
