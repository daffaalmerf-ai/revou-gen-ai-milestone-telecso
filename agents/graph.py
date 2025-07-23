from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from typing import Annotated, List, Union
from agents import (
    oracle_cache_agent,
    elastic_agent,
    vector_agent,
    rerank_agent,
    price_lookup_agent,
    po_extractor_agent,
    enrich_order_agent,
    modify_order_agent
)

# Define the state
class ProductState(TypedDict, total=False):
    query: str
    embedding: List[float]
    oracle_result: dict
    elastic_result: dict
    milvus_result: dict
    reranked_result: dict
    final_result: dict
    price: float
    po_file: bytes
    extracted_order: dict
    enriched_order: dict
    user_modified_order: dict

# Create the graph
graph = StateGraph(ProductState)

# Add product agents
graph.add_node("oracle", oracle_cache_agent)
graph.add_node("elastic", elastic_agent)
graph.add_node("milvus", vector_agent)
graph.add_node("rerank", rerank_agent)
graph.add_node("price", price_lookup_agent)

# Add PO upload agents
graph.add_node("extract_po", po_extractor_agent)
graph.add_node("enrich_order", enrich_order_agent)
graph.add_node("modify_order", modify_order_agent)

# Define conditional logic
def route_product(state: ProductState):
    if state.get("query"):
        return "oracle"
    elif state.get("po_file"):
        return "extract_po"
    return END

def route_product_match(state: ProductState):
    if not state.get("oracle_result"):
        return "elastic"
    return "rerank"

def route_elastic(state: ProductState):
    if not state.get("elastic_result"):
        return "milvus"
    return "rerank"

def route_order(state: ProductState):
    return "enrich_order"

def route_modify_order(state: ProductState):
    return "modify_order"

# Edges for search query flow
graph.set_entry_point(route_product)
graph.add_edge("oracle", route_product_match)
graph.add_edge("elastic", route_elastic)
graph.add_edge("milvus", "rerank")
graph.add_edge("rerank", "price")
graph.add_edge("price", END)

# Edges for PO flow
graph.add_edge("extract_po", route_order)
graph.add_edge("enrich_order", route_modify_order)
graph.add_edge("modify_order", END)

# Compile
graph.set_finish_point(END)
product_graph = graph.compile()

# .add_node("oracle_cache_agent", oracle_cache_agent)
#     .add_node("elasticsearch_agent", elasticsearch_agent)
#     .add_node("milvus_agent", milvus_agent)
#     .add_node("gpt_reranker", gpt_reranker)
#     .add_conditional_edges(
#         "oracle_cache_agent",
#         tools_condition,
#         {
#             END: END,
#             "fallback": "elasticsearch_agent"
#         }
#     )
#     .add_conditional_edges(
#         "elasticsearch_agent",
#         tools_condition,
#         {
#             END: END,
#             "milvus_fallback": "milvus_agent"
#         }
#     )
#     .add_conditional_edges(
#         "milvus_agent",
#         tools_condition,
#         {
#             "rerank": "gpt_reranker",
#             END: END
#         }
#     )
#     .add_edge("gpt_reranker", END)
#     .set_entry_point("oracle_cache_agent")
#     .compile()