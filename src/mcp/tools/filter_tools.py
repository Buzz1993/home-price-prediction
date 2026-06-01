import json
from rapidfuzz import process
import pandas as pd

from src.utils.search_metadata import (
    get_filter_schema,
    build_search_metadata
)

# Load data once
master_df = pd.read_csv(
    "data/cleaned/final_cleaned_rec_data.csv"
)

SEARCH_METADATA = build_search_metadata(
    master_df
)


def get_available_filters() -> str:
    """
    Returns searchable columns and counts.
    """

    schema = get_filter_schema(
        SEARCH_METADATA
    )

    return json.dumps(
        schema,
        indent=2
    )


def search_filter_values(
    column: str,
    query: str,
    top_k: int = 20
) -> str:
    """
    Search values inside a column.
    """

    if column not in SEARCH_METADATA:

        return json.dumps(
            {
                "error":
                f"Unknown column: {column}"
            },
            indent=2
        )

    values = SEARCH_METADATA[column]

    matches = process.extract(
        query,
        values,
        limit=top_k
    )

    results = []

    for value, score, _ in matches:

        if score >= 60:

            results.append(
                {
                    "value": value,
                    "score": score
                }
            )

    return json.dumps(
        {
            "column": column,
            "query": query,
            "matches": results
        },
        indent=2
    )