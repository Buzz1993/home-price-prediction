# ===============================
# workflow.py
# ===============================

from langgraph.graph import StateGraph, START, END

from src.graph.state import PropertyState

from src.graph.nodes.search_node import search_node
from src.graph.nodes.comparison_node import comparison_node
from src.graph.nodes.explanation_node import explanation_node

from src.graph.nodes.memory_node import memory_node
from src.graph.nodes.router_node import router_node
from src.graph.nodes.valuation_node import valuation_node
from src.graph.nodes.negotiation_node import negotiation_node
from src.graph.nodes.general_chat_node import general_chat_node
from src.graph.nodes.prediction_node import prediction_node
from src.graph.nodes.rental_node import rental_node

def route_decision(state):

    return state["route"]


# =========================================
# SEARCH GRAPH
# =========================================
search_builder = StateGraph(PropertyState)

#nodes
search_builder.add_node("search", search_node)

#edges
search_builder.add_edge(START, "search")
search_builder.add_edge("search", END)

search_graph = search_builder.compile()


# =========================================
# COMPARISON GRAPH
# =========================================
comparison_builder = StateGraph(PropertyState)

#nodes
comparison_builder.add_node("comparison", comparison_node)
comparison_builder.add_node("explanation", explanation_node)

#edges
comparison_builder.add_edge(START, "comparison")
comparison_builder.add_edge("comparison", "explanation")
comparison_builder.add_edge("explanation", END)

comparison_graph = comparison_builder.compile()


# =========================================
# CHAT GRAPH
# =========================================
chat_builder = StateGraph(PropertyState)

# NODES
chat_builder.add_node("memory", memory_node)

chat_builder.add_node("router", router_node)

chat_builder.add_node("valuation", valuation_node)

chat_builder.add_node("prediction", prediction_node)

chat_builder.add_node("negotiation", negotiation_node)

chat_builder.add_node("rental", rental_node)

chat_builder.add_node("general", general_chat_node)

# edges
chat_builder.add_edge(START, "memory")
chat_builder.add_edge("memory", "router")

# CONDITIONAL ROUTING
chat_builder.add_conditional_edges(
    "router",   # "router" means after router_node executes,
                # LangGraph performs conditional routing using
                # route_decision(state). route_decision reads
                # state["route"] and LangGraph matches that
                # route value with the mapping below to execute
                # the corresponding node.
    route_decision,
    {
        "valuation": "valuation",

        "prediction": "prediction",

        "negotiation": "negotiation",

        "rental": "rental",

        "general": "general"
    }
)

# ENDING
chat_builder.add_edge("valuation", END)

chat_builder.add_edge("prediction", END)

chat_builder.add_edge("negotiation", END)

chat_builder.add_edge("rental", END)

chat_builder.add_edge("general", END)

# COMPILE
chat_graph = chat_builder.compile()