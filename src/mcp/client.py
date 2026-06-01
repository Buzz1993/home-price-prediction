# ===============================
# client.py
# ===============================

import asyncio

from fastmcp import Client


async def main():

    # 1. Start server.py
    # 2. Create MCP connection
    # 3. Connect client ↔ server
    async with Client(
        "src/mcp/server.py"
    ) as client:

        # Send a request to the MCP server to execute the search_properties tool with filters={"city": "thane"}.
        result = await client.call_tool(
            "search_properties",
            {
                "city": "thane",
                "bhk": 2
            }
        )

        print(result) # result is the response from the server after executing the search_properties tool.


asyncio.run(main())

# Example:
#
# Client sends:
#     search_properties(city="thane")
#
# Server receives request
#     ↓
# Executes search_properties tool
#     ↓
# Calls run_search_pipeline()
#     ↓
# Returns result to client