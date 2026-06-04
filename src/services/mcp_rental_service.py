#================================
# mcp_rental_service.py
#================================

import pandas as pd
from src.agents.rental_agent import (
    run_rental_agent
)

from src.data.data_store import master_df


def run_mcp_rental(
    property_ids: list[str]
):

    selected_df = master_df[
        master_df["id"].astype(str).isin(
            [str(x) for x in property_ids]
        )
    ].copy()

    rental_df = run_rental_agent(
        selected_df
    )

    return rental_df