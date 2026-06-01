#main.py

import pandas as pd

from mcp.server.fastmcp import FastMCP

from tools.comparison_tools import (
    register_comparison_tools
)

from tools.filter_tools import (
    register_filter_tools
)

from src.utils.search_metadata import (
    build_search_metadata
)

# =====================================
# MCP SERVER
# =====================================
mcp = FastMCP(
    "Real Estate MCP Server"
)

# =====================================
# LOAD DATA
# =====================================
master_df = pd.read_csv(
    "data/cleaned/final_cleaned_rec_data.csv"
)

# =====================================
# BUILD SEARCH METADATA
# =====================================
SEARCH_METADATA = build_search_metadata(
    master_df
)

# =====================================
# REGISTER TOOLS
# =====================================
register_filter_tools(
    mcp,
    SEARCH_METADATA
)

register_comparison_tools(
    mcp,
    master_df
)

# =====================================
# RUN SERVER
# =====================================
if __name__ == "__main__":
    mcp.run()