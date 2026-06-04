#===============================
# data_store.py
#===============================

import pandas as pd
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[2]

master_df = pd.read_csv(
    ROOT_PATH / "data" / "cleaned" / "final_combined_mcp_data.csv"
)