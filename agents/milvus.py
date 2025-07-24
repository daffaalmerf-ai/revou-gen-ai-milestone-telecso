import os
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv(override=True)

model = init_chat_model("gpt-4.1-mini", model_provider="openai")

def connect_milvus(embedding_model, milvus_db, milvus_host, milvus_port, milvus_collection):
    embedding_model = OpenAIEmbeddings(model=embedding_model)

    vector_store = Milvus(
        embedding_function=embedding_model,
        connection_args={
            "uri": f"http://{milvus_host}:{milvus_port}",
            "db_name": milvus_db
        },
        collection_name=milvus_collection,
    )

    return vector_store

milvus_db = os.getenv("MILVUS_DB")
milvus_host = os.getenv("MILVUS_HOST")
milvus_port = os.getenv("MILVUS_PORT")
milvus_collection = os.getenv("MILVUS_COLLECTION")

vector_store = connect_milvus("text-embedding-3-small", milvus_db, milvus_host, milvus_port, milvus_collection)
@tool(response_format="content_and_artifact")
def retrieve_similar_product(query: str):
    """Retrieve similar product based on user query."""
    retrieved_docs = vector_store.similarity_search(query, k=5)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def query_or_respond_similar_product(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    model_with_tools = model.bind_tools([retrieve_similar_product])
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tools_similar_product = ToolNode([retrieve_similar_product], name="tools_similar_product")

def generate_similar_product(state: MessagesState):
    """Generate answer."""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    response = None

    if len(tool_messages) > 0 and len(tool_messages[-1].content):

        docs_content = "\n\n".join(
            doc.page_content
            for m in tool_messages
            if hasattr(m, "artifact") and isinstance(m.artifact, list)
            for doc in m.artifact
            if isinstance(doc, Document)
        )

        system_message_content = (
            "You are a Customer Service Officer (CSO) assigned to recommend similar products based on a given product."
            "You have access to detailed information for each product, including its description, indication, usage, how to use, dosage, and side effects."
            "If any of these fields are missing (e.g., contain NaN), reduce the confidence score of the recommendation."
            "If no sufficiently similar product is found, respond with 'Maaf, saya belum bisa menemukan produk serupa untuk saat ini.'"
            "Your response must be concise, and for each recommended product, provide a brief description for each of them."
            "Use a natural and polite tone in your response."
            "\n\n"
            "Use the following docs:"
            f"{docs_content}"
        )

        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]

        prompt = [SystemMessage(system_message_content)] + conversation_messages

        response = model.invoke(prompt)

    else:
        response = state["messages"][-1]

    return {"messages": [response]}


graph = (
    StateGraph(MessagesState)
    .add_node(query_or_respond_similar_product)
    .add_node(tools_similar_product)
    .add_node(generate_similar_product)
    .set_entry_point("query_or_respond_similar_product")
    .add_conditional_edges(
    "query_or_respond_similar_product",
        tools_condition,
        {END: END, "tools": "tools_similar_product"},
    )
    .add_edge("tools_similar_product", "generate_similar_product")
    .add_edge("generate_similar_product", END)
    .compile(name="MilvusAgent")
)