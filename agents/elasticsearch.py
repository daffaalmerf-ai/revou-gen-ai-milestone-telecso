from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import ToolNode, tools_condition
from typing import List

import os
import json
import requests

load_dotenv(override=True)

llm = init_chat_model("gpt-4.1-mini", model_provider="openai")

@tool("query_product_elasticsearch", parse_docstring=True)
def query_product_elasticsearch(product_names: List[str]):
    """
    Search for similar product names using Elasticsearch and return the top match and score.

    Args:
        product_names: List of names or partial names of the product provided by the customer.

    Returns:
        Array of products with _index, _type, _id, _score, and _source. Possible to return
        empty array.
    """
    hits_results = []
    for product_name in product_names:
        url = os.environ["ELASTIC_URL"].rstrip("/") + "/_search"
        auth = (os.environ["ELASTIC_USERNAME"], os.environ["ELASTIC_PASSWORD"])

        query_elastic = {
            "query": {
                "dis_max": {
                    "queries": [
                        {"match": {"name": {"query": product_name}}},
                        {"match": {"name_autocomplete": {"query": product_name}}},
                        {"match": {"name_synonym": {"query": product_name}}}
                    ]
                }
            }
        }
        response = requests.get(url, auth=auth, json=query_elastic)
        response.raise_for_status()
        hits = json.dumps(response.json().get("hits", {}).get("hits", []))
        hits_results.append({product_name: hits})
    return hits_results

def query_or_respond_elastic(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([query_product_elasticsearch])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tools_elastic = ToolNode([query_product_elasticsearch], name="tools_elastic")

def generate_elastic_code(state: MessagesState):
    """Generate answer."""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(msg.content for msg in tool_messages)

    system_message_content = (
        "You are an expert Customer Service Officer (CSO) who understands how to map the customer's product name "
        "to the company's official product name. Use Elasticsearch results to determine the best match.\n\n"
    
        "Instructions:\n"
        "- If the highest `_score` in the Elasticsearch results is less than 25, you must also perform Jaccard similarity "
        "to justify your best choice.\n"
        "- Return your final answer in this format:\n"
        "  { 'product_code': _source[i].code }\n"
        "  where `i` is the index of the best-matching result.\n\n"
    
        "Here are some examples of product mappings:\n"
        "- Tride Tablet 5000 unit (vitamin D) → TRIDE 5000IU BOX 10 STR @ 6 KAP\n"
        "- Tensivask Tablet 5 MG (amlodipine) → TENSIVASK 5MG @50\n"
        "- (G) REG 3 PROPRANOLOL TABLET 10 MG DEXA → PROPRANOLOL 10MG @100(DX)\n\n"
    
        "Store the results in the same manner of product key-value pair {product_name}:{best_match_product_name}"
    
        f"Elasticsearch results:\n{docs_content}\n"
    )

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]

    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = llm.invoke(prompt)

    return {"messages": [response]}

graph = (
    StateGraph(MessagesState)
    .add_node(query_or_respond_elastic)
    .add_node(tools_elastic)
    .add_node(generate_elastic_code)
    .set_entry_point("query_or_respond_elastic")
    .add_conditional_edges(
        "query_or_respond_elastic",
        tools_condition,
        {END: END, "tools": "tools_elastic"}
    )
    .add_edge("tools_elastic", "generate_elastic_code")
    .add_edge("generate_elastic_code", END)
    .compile(name="ElasticAgent")
)
