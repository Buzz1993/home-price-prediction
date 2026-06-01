# src/mcp/tools/comparison_tools.py

import json

import pandas as pd

from src.services.comparison_service import (
    run_comparison
)

# load once
master_df = pd.read_csv(
    "data/cleaned/final_cleaned_rec_data.csv"
)


def compare_properties(
    property_ids: list[str]
) -> str:
    """
    Compare multiple properties and
    return investment ranking.
    """

    selected_df = master_df[
        master_df["id"].astype(str).isin(
            [str(x) for x in property_ids]
        )
    ].copy()

    if len(selected_df) < 2:

        return json.dumps(
            {
                "error":
                "Need at least 2 properties"
            },
            indent=2
        )

    raw_df, compare_df = run_comparison(
        selected_df
    )

    compare_df = compare_df.sort_values(
        "overall_score",
        ascending=False
    )

    winner = compare_df.iloc[0]

    result = {
        "winner": {
            "id": str(
                winner["id"]
            ),
            "overall_score": float(
                winner["overall_score"]
            ),
            "verdict": str(
                winner["verdict"]
            ),
            "comparison_reason": str(
                winner["comparison_reason"]
            )
        },
        "rankings": compare_df[
            [
                "id",
                "overall_score",
                "verdict",
                "comparison_reason"
            ]
        ].to_dict(
            orient="records"
        )
    }

    return json.dumps(
        result,
        indent=2,
        default=str
    )