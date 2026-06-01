#advisor_tool.py

from src.agents.advisor_agent import (
    run_advisor_agent
)

import pandas as pd


def investment_advisor():

    raw_df = pd.read_csv(
        "comparison_raw.csv"
    )

    compare_df = pd.read_csv(
        "comparison_result.csv"
    )

    advisor_df = run_advisor_agent(
        raw_df
    )

    advisor_df = advisor_df.merge(
        compare_df[
            [
                "id",
                "overall_score",
                "verdict"
            ]
        ],
        on="id"
    )

    winner = advisor_df.sort_values(
        "overall_score",
        ascending=False
    ).iloc[0]

    return {
        "recommended_property":
            winner["id"],

        "verdict":
            winner["verdict"],

        "positives":
            winner["positives"],

        "risks":
            winner["risks"]
    }