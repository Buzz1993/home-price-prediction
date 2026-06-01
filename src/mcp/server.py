# ===============================
# server.py
# ===============================
import sys

if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8")

if sys.stderr:
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

# Create MCP Server instance.
# All tools decorated with @mcp.tool
# will be registered on this server.
mcp = FastMCP(
    name="Property AI MCP Server"
) # name of the mcp server we keep here as "Property AI MCP Server" 

# created mcp tools in separate files in src/mcp/tools/ and we import those tools here in server.py 
# =====================================
# TOOLS
# =====================================
from src.mcp.tools.search_tool import (
    search_properties
)

from src.mcp.tools.comparison_tools import (
    compare_properties
)

from src.mcp.tools.prediction_tool import (
    predict_price
)

from src.mcp.tools.advisor_tool import (
    investment_advisor
)

from src.mcp.tools.filter_tools import (
    get_available_filters,
    search_filter_values
)

# =====================================
# REGISTER TOOLS
# =====================================
mcp.tool(
    search_properties
)

mcp.tool(
    compare_properties
)

mcp.tool(
    predict_price
)

mcp.tool(
    investment_advisor
)

mcp.tool(
    get_available_filters
)

mcp.tool(
    search_filter_values
)

# Start MCP Server.
# Server waits for incoming requests
# from MCP clients and routes them
# to the appropriate registered tool.
if __name__ == "__main__":
    mcp.run() #this use to run the server