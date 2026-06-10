# """
# Production-Ready FastMCP Server Bridge.
# Exposes clean interfaces for search, batch extraction, evaluation, metadata mapping, and context-steering paths.
# """
# import os
# import sys
# from pathlib import Path

# ROOT_DIR = Path(__file__).resolve().parents[2]
# if str(ROOT_DIR) not in sys.path:
#     sys.path.insert(0, str(ROOT_DIR))

# from fastmcp import FastMCP
# import src.mcp.tools.property_tools as tools
# from src.core.search_registry import CACHED_SEARCH_METADATA

# mcp = FastMCP(
#     "Real Estate Analytical Intent Engine",
#     dependencies=["pandas", "numpy", "rank_bm25", "requests"]
# )

# # =====================================================================
# # 🛠️ PROTOCOL TOOL EXPOSURE ROUTING LAYERS
# # =====================================================================

# @mcp.tool()
# def search_properties(bhk: str = None, amenities: str = None, location: str = None, limit: int = 5) -> list[dict]:
#     """
#     Search real estate inventory via BM25 token keywords and optional exact BHK formatting constraints.
#     Returns targeted whitelisted column schemas to maximize system-wide token efficiency.
#     """
#     return tools.search_properties(bhk=bhk, amenities=amenities, location=location, limit=limit)


# @mcp.tool()
# def get_property_details(property_id: str) -> dict:
#     """
#     Retrieve clean whitelisted specifications for an individual property using its unique ID string.
#     Use this route to parse all layouts, specs, pricing, and structural asset indicators.
#     """
#     return tools.get_property_details(property_id)


# @mcp.tool()
# def get_properties_by_ids(property_ids: list[str]) -> list[dict]:
#     """
#     Batch extraction endpoint. Resolves details for multiple property IDs simultaneously.
#     Reduces system latency by resolving an array of hits down to core whitelisted specs in one execution loop.
#     """
#     return tools.get_properties_by_ids(property_ids)


# @mcp.tool()
# def compare_properties(property_ids: list[str]) -> dict:
#     """
#     Compare multiple real estate assets side-by-side using normalized stacking scores.
#     Requires a minimum list of 2 or more target property IDs.
#     """
#     return tools.compare_properties(property_ids)


# @mcp.tool()
# def get_rental_analysis(property_ids: list[str]) -> list[dict]:
#     """Extract micro-rental income generation metrics, estimates, and rental yields for an array of property IDs."""
#     return tools.get_rental_analysis(property_ids)


# @mcp.tool()
# def get_price_prediction(property_ids: list[str]) -> list[dict]:
#     """Invokes external ML core pipeline inference models to forecast market valuation differences."""
#     return tools.get_price_prediction(property_ids)


# @mcp.tool()
# def get_negotiation_strategy(property_ids: list[str]) -> list[dict]:
#     """Generates tailored strategic talking points, suggested percentage discount buffers, and target prices."""
#     return tools.get_negotiation_strategy(property_ids)


# @mcp.tool()
# def get_valuation_analysis(property_ids: list[str]) -> list[dict]:
#     """Evaluates real-time pricing distribution margins to flag whether a property is overpriced, undervalued, or fair."""
#     return tools.get_valuation_analysis(property_ids)


# @mcp.tool()
# def get_investment_advice(property_ids: list[str]) -> list[dict]:
#     """Orchestrates macro portfolio suitability reviews matching property attributes against specific buyer profiles."""
#     return tools.get_investment_advice(property_ids)


# @mcp.tool()
# def clear_property_analysis_cache() -> dict:
#     """Flushes out long-running property enrichment memory fragments across thread lines."""
#     return tools.clear_property_analysis_cache()


# # =====================================================================
# # 📊 PROTOCOL RESOURCE LAYER (ZERO RUNTIME DISK READ LATENCY)
# # =====================================================================

# @mcp.resource("real-estate://metadata")
# def get_search_metadata() -> dict:
#     """
#     Exposes pristine compiled dictionary lookup keys for system-wide validation fields straight out of RAM cache.
#     Downstream LLM engines parse this configuration schema instantly to formulate correct criteria mappings.
#     """
#     return CACHED_SEARCH_METADATA


# # =====================================================================
# # 🎯 AUTOMATED STEERING PROMPT PIPELINES
# # =====================================================================

# @mcp.prompt()
# def property_investment_advisor() -> str:
#     """
#     System steering profile template context that converts downstream LLMs into institutional investment advisors.
#     """
#     return """You are a high-conviction institutional real estate investment advisor specializing in high-yield assets.
#     Your operational sequence must follow this strict workflow protocol:
    
#     1. Read and review allowable lookup entries via the primary system resource template parameters at 'real-estate://metadata'.
#     2. Dynamically execute broad intent matching filtering lookups via the 'search_properties' tool matching client requests.
#     3. Isolate the unique property IDs returned from the discovery step and invoke 'get_properties_by_ids' to fetch all specifications in a single batch pass.
#     4. Pass the extracted entities downstream through 'compare_properties' and 'get_investment_advice' scoring layers.
#     5. Present comprehensive responses formatting parameters cleanly using Indian Rupee (₹) denomination specifications. Highlight systematic asset risks explicitly alongside gross yield projections.
#     """


# if __name__ == "__main__":
#     mcp.run()

#==========================================================================================================================================================================

# """
# Production-Ready FastMCP Server Bridge.
# Exposes clean interfaces for search, batch extraction, evaluation, metadata mapping, and context-steering paths.
# """
# import os
# import sys
# from pathlib import Path

# ROOT_DIR = Path(__file__).resolve().parents[2]
# if str(ROOT_DIR) not in sys.path:
#     sys.path.insert(0, str(ROOT_DIR))

# from fastmcp import FastMCP
# import src.mcp.tools.property_tools as tools
# from src.core.search_registry import CACHED_SEARCH_METADATA

# mcp = FastMCP(
#     "Real Estate Analytical Intent Engine"
# )

# # =====================================================================
# # 🛠️ PROTOCOL TOOL EXPOSURE ROUTING LAYERS
# # =====================================================================

# @mcp.tool()
# def search_properties(bhk: str = None, amenities: str = None, location: str = None, limit: int = 5) -> list[dict]:
#     """
#     Search real estate inventory via BM25 token keywords and optional exact BHK formatting constraints.
#     Returns targeted whitelisted column schemas to maximize system-wide token efficiency.
#     """
#     return tools.search_properties(bhk=bhk, amenities=amenities, location=location, limit=limit)


# @mcp.tool()
# def get_property_details(property_id: str) -> dict:
#     """
#     Retrieve clean whitelisted specifications for an individual property using its unique ID string.
#     Use this route to parse all layouts, specs, pricing, and structural asset indicators.
#     """
#     return tools.get_property_details(property_id)


# @mcp.tool()
# def get_properties_by_ids(property_ids: list[str]) -> list[dict]:
#     """
#     Batch extraction endpoint. Resolves details for multiple property IDs simultaneously.
#     Reduces system latency by resolving an array of hits down to core whitelisted specs in one execution loop.
#     """
#     return tools.get_properties_by_ids(property_ids)


# @mcp.tool()
# def compare_properties(property_ids: list[str]) -> dict:
#     """
#     Compare multiple real estate assets side-by-side using normalized stacking scores.
#     Requires a minimum list of 2 or more target property IDs.
#     """
#     return tools.compare_properties(property_ids)


# @mcp.tool()
# def get_rental_analysis(property_ids: list[str]) -> list[dict]:
#     """Extract micro-rental income generation metrics, estimates, and rental yields for an array of property IDs."""
#     return tools.get_rental_analysis(property_ids)


# @mcp.tool()
# def get_price_prediction(property_ids: list[str]) -> list[dict]:
#     """Invokes external ML core pipeline inference models to forecast market valuation differences."""
#     return tools.get_price_prediction(property_ids)


# @mcp.tool()
# def get_negotiation_strategy(property_ids: list[str]) -> list[dict]:
#     """Generates tailored strategic talking points, suggested percentage discount buffers, and target prices."""
#     return tools.get_negotiation_strategy(property_ids)


# @mcp.tool()
# def get_valuation_analysis(property_ids: list[str]) -> list[dict]:
#     """Evaluates real-time pricing distribution margins to flag whether a property is overpriced, undervalued, or fair."""
#     return tools.get_valuation_analysis(property_ids)


# @mcp.tool()
# def get_investment_advice(property_ids: list[str]) -> list[dict]:
#     """Orchestrates macro portfolio suitability reviews matching property attributes against specific buyer profiles."""
#     return tools.get_investment_advice(property_ids)


# @mcp.tool()
# def clear_property_analysis_cache() -> dict:
#     """Flushes out long-running property enrichment memory fragments across thread lines."""
#     return tools.clear_property_analysis_cache()


# # =====================================================================
# # 📊 PROTOCOL RESOURCE LAYER (ZERO RUNTIME DISK READ LATENCY)
# # =====================================================================

# @mcp.resource("real-estate://metadata")
# def get_search_metadata() -> dict:
#     """
#     Exposes pristine compiled dictionary lookup keys for system-wide validation fields straight out of RAM cache.
#     Downstream LLM engines parse this configuration schema instantly to formulate correct criteria mappings.
#     """
#     return CACHED_SEARCH_METADATA


# # =====================================================================
# # 🎯 AUTOMATED STEERING PROMPT PIPELINES
# # =====================================================================

# @mcp.prompt()
# def property_investment_advisor() -> str:
#     """
#     System steering profile template context that converts downstream LLMs into institutional investment advisors.
#     """
#     return """You are a high-conviction institutional real estate investment advisor specializing in high-yield assets.
#     Your operational sequence must follow this strict workflow protocol:
    
#     1. Read and review allowable lookup entries via the primary system resource template parameters at 'real-estate://metadata'.
#     2. Dynamically execute broad intent matching filtering lookups via the 'search_properties' tool matching client requests.
#     3. Isolate the unique property IDs returned from the discovery step and invoke 'get_properties_by_ids' to fetch all specifications in a single batch pass.
#     4. Pass the extracted entities downstream through 'compare_properties' and 'get_investment_advice' scoring layers.
#     5. Present comprehensive responses formatting parameters cleanly using Indian Rupee (₹) denomination specifications. Highlight systematic asset risks explicitly alongside gross yield projections.
#     """ 

#============================================================================================================================================================================
# ==============================
# src/mcp/server.py
# ==============================

"""
Production-Ready FastMCP Server Bridge.
Exposes clean interfaces for search, batch extraction, evaluation, metadata mapping, and context-steering paths.
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastmcp import FastMCP
import src.mcp.tools.property_tools as tools
from src.core.search_registry import CACHED_SEARCH_METADATA

mcp = FastMCP(
    "Real Estate Analytical Intent Engine"
)

# =====================================================================
# 🛠️ PROTOCOL TOOL EXPOSURE ROUTING LAYERS
# =====================================================================

@mcp.tool()
def search_properties(bhk: str = None, amenities: str = None, location: str = None, limit: int = 5) -> list[dict]:
    """
    Search real estate inventory via BM25 token keywords and optional exact BHK formatting constraints.
    Returns targeted whitelisted column schemas to maximize system-wide token efficiency.
    """
    return tools.search_properties(bhk=bhk, amenities=amenities, location=location, limit=limit)


@mcp.tool()
def get_property_details(property_id: str) -> dict:
    """
    Retrieve clean whitelisted specifications for an individual property using its unique ID string.
    Use this route to parse all layouts, specs, pricing, and structural asset indicators.
    """
    return tools.get_property_details(property_id)


@mcp.tool()
def get_properties_by_ids(property_ids: list[str]) -> list[dict]:
    """
    Batch extraction endpoint. Resolves details for multiple property IDs simultaneously.
    Reduces system latency by resolving an array of hits down to core whitelisted specs in one execution loop.
    """
    return tools.get_properties_by_ids(property_ids)


@mcp.tool()
def compare_properties(property_ids: list[str]) -> dict:
    """
    Compare multiple real estate assets side-by-side using normalized stacking scores.
    Requires a minimum list of 2 or more target property IDs.
    """
    return tools.compare_properties(property_ids)


@mcp.tool()
def get_rental_analysis(property_ids: list[str]) -> list[dict]:
    """Extract micro-rental income generation metrics, estimates, and rental yields for an array of property IDs."""
    return tools.get_rental_analysis(property_ids)


@mcp.tool()
def get_price_prediction(property_ids: list[str]) -> list[dict]:
    """Invokes external ML core pipeline inference models to forecast market valuation differences."""
    return tools.get_price_prediction(property_ids)


@mcp.tool()
def get_negotiation_strategy(property_ids: list[str]) -> list[dict]:
    """Generates tailored strategic talking points, suggested percentage discount buffers, and target prices."""
    return tools.get_negotiation_strategy(property_ids)


@mcp.tool()
def get_valuation_analysis(property_ids: list[str]) -> list[dict]:
    """Evaluates real-time pricing distribution margins to flag whether a property is overpriced, undervalued, or fair."""
    return tools.get_valuation_analysis(property_ids)


@mcp.tool()
def get_investment_advice(property_ids: list[str]) -> list[dict]:
    """Orchestrates macro portfolio suitability reviews matching property attributes against specific buyer profiles."""
    return tools.get_investment_advice(property_ids)


@mcp.tool()
def clear_property_analysis_cache() -> dict:
    """Flushes out long-running property enrichment memory fragments across thread lines."""
    return tools.clear_property_analysis_cache()

@mcp.tool()
def send_property_report(
    phone_number: str,
    report: str
) -> dict:
    """
    Sends property report to n8n workflow.
    """
    return tools.send_property_report(
        phone_number=phone_number,
        report=report
    )


# =====================================================================
# 📊 PROTOCOL RESOURCE LAYER (ZERO RUNTIME DISK READ LATENCY)
# =====================================================================

@mcp.resource("real-estate://metadata")
def get_search_metadata() -> dict:
    """
    Exposes pristine compiled dictionary lookup keys for system-wide validation fields straight out of RAM cache.
    Downstream LLM engines parse this configuration schema instantly to formulate correct criteria mappings.
    """
    return CACHED_SEARCH_METADATA


# =====================================================================
# 🎯 AUTOMATED STEERING PROMPT PIPELINES
# =====================================================================

@mcp.prompt()
def property_investment_advisor() -> str:
    """
    System steering profile template context that converts downstream LLMs into institutional investment advisors.
    """
    return """You are a high-conviction institutional real estate investment advisor specializing in high-yield assets.
    Your operational sequence must follow this strict workflow protocol:
    
    1. Read and review allowable lookup entries via the primary system resource template parameters at 'real-estate://metadata'.
    2. Dynamically execute broad intent matching filtering lookups via the 'search_properties' tool matching client requests.
    3. Isolate the unique property IDs returned from the discovery step and invoke 'get_properties_by_ids' to fetch all specifications in a single batch pass.
    4. Pass the extracted entities downstream through 'compare_properties' and 'get_investment_advice' scoring layers.
    5. Present comprehensive responses formatting parameters cleanly using Indian Rupee (₹) denomination specifications. Highlight systematic asset risks explicitly alongside gross yield projections.
    """ 




