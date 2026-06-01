#==============================
# generate_search_metadata.py
# ===============================
# Generate a searchable metadata dictionary from property dataset columns and save it as search_metadata.json for MCP filtering and natural-language matching.

import json
import pandas as pd


KEEP_COLUMNS = [
    "bhk_type",
    "flooring",
    "construction",
    "ownership",
    "builder",
    "project_name",
    "furnish",
    "city",
    "location",
    "property_type",
    "status",
    "extra_rooms",
    "facing",
    "overlooking",
    "transportation_hubs_clean"
]


def extract_values(series):
    """
    Extract unique values from a column.
    Handles:
    - comma separated values
    - pipe separated values
    - semicolon separated values
    """

    values = set()

    for item in (
        series.dropna()
        .astype(str)
    ):

        item = item.strip()

        if not item:
            continue

        separators = [
            ",",
            "|",
            ";"
        ]

        found_separator = False

        for sep in separators:

            if sep in item:

                found_separator = True

                for value in item.split(sep):

                    value = value.strip()

                    if value:
                        values.add(value)

        if not found_separator:
            values.add(item)

    return sorted(values)


def build_search_metadata(df):

    metadata = {}

    for col in KEEP_COLUMNS:

        if col not in df.columns:
            continue

        values = extract_values(
            df[col]
        )

        metadata[col] = values

        print(
            f"{col}: {len(values)} values"
        )

    return metadata


def main():

    ROOT_DIR = Path(__file__).resolve().parents[1]

    df = pd.read_csv(
        ROOT_DIR /
        "data" /
        "cleaned" /
        "final_cleaned_rec_data.csv"
    )

    metadata = build_search_metadata(
        df
    )

    output_file = (
        ROOT_DIR /
        "data" /
        "search_metadata.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nSaved metadata to: {output_file}"
    )


if __name__ == "__main__":
    main()