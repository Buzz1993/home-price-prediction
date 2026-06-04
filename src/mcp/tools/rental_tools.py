#================================
# rental_tools.py
#================================

import json

from src.services.mcp_rental_service import run_mcp_rental


def get_rental_analysis(
    property_ids
):

    rental_df = run_mcp_rental(
        property_ids
    )

    return json.dumps(
        rental_df.to_dict(
            orient="records"
        ),
        indent=2,
        default=str
    )