from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from typing import Annotated
from agents.elasticsearch import graph as ElasticAgent 

import os
import cx_Oracle
import re

load_dotenv(override=True)

model = init_chat_model("gpt-4.1-mini", model_provider= "openai")

class DBGraphState(MessagesState):
    user_question: Annotated[str, "User question that must be answered by querying the database"]
    customer_id_reference: Annotated[int, "User's customer id reference as unique identifier"]
    fallback_elastic: Annotated[bool, "Determine if fallback to elasticsearch is already hit"]

DB_CONFIG_ORACLE = {
    "HOST": os.environ["ORACLE_HOST"],
    "PORT": os.environ["ORACLE_PORT"],
    "SERVICE_NAME": os.environ["ORACLE_SERVICE_NAME"],
    "USERNAME": os.environ["ORACLE_USERNAME"],
    "PASSWORD": os.environ["ORACLE_PASSWORD"]
}

TABLES_ALLOWED_ORACLE = [
    { "owner": "AAM_DWH", "table": "DIM_CUSTOMER"},
    { "owner": "AAM_DWH", "table": "DIM_PRODUCT"},
    { "owner": "MISDSAAM", "table": "AAM_PRODUCT_SUGGEST_CSO"}
]

def oracle_set_connection(project_name, db_config):
    try:
        dsn = cx_Oracle.makedsn(db_config['HOST'], db_config['PORT'], service_name=db_config['SERVICE_NAME'])
        conn = cx_Oracle.connect(db_config['USERNAME'], db_config['PASSWORD'], dsn)
        return conn
    except Exception as e:
        print(f'Exception caught when initializing Oracle engine in {project_name}: {e}')
    return None

conn = oracle_set_connection("DEMO-TELECSO", DB_CONFIG_ORACLE)
cursor = conn.cursor()

@tool("query_executor", parse_docstring=True)
def query_executor(query: str):
    """
        Execute select queries to Oracle database with the provided database configurations.

        Args:
        query = select statement that will be executed
        
        Return:
        Stringified result(s) of the query.
    """
    try:
        query = re.sub(';', '', query)
        cursor.execute(query)
        result = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        results_arr = [dict(zip(columns, row)) for row in result]
        return results_arr
    except Exception as e:
        print("Exception caught")
        print(e)

tools_oracle_query = ToolNode([query_executor], name="tools_oracle_query")

def query_generator(state: DBGraphState):
    dialect = "Oracle"
    user_question = state['user_question']
    customer_id_reference = state['customer_id_reference']
    top_k = 10
    instruction = SystemMessage(content=f"""You are an expert data analyst designed to support Customer Service Officers (CSOs) in retrieving *product-related* information from a SQL database using {dialect} dialect. Your role is strictly limited to answering questions about product **price** or **stock** only. You must reject any other types of questions such as customer addresses, transaction history, or anything unrelated to price or stock.
                                Your behavior must follow the two-step format:
                                1. Create a **syntactically correct and safe SQL query** to answer the question using the input field `customer_id_reference` as a filter.
                                2. Return the plain SQL query strictly with the {dialect} dialect

                                User question: {user_question}
                                Customer id reference: {customer_id_reference}

                                ====================
                                ## RULES:
                                # ====================
                                # - ✅ Only answer questions about product **price** or **stock**.
                                # - ❌ DO NOT generate queries for unrelated questions (e.g., address, phone number, transaction, etc).
                                # - ❌ DO NOT include any DML or DDL operations (INSERT, UPDATE, DELETE, DROP, etc).
                                # - ✅ Always use the provided `customer_id_reference` to determine customer context.
                                # - ✅ Always restrict results to the most recent or relevant row using ROW_NUMBER() or `MAX(CUSTOMER_KEY)`.
                                # - ✅ Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
                                # - ✅ Limit the number of returned rows and iterate if multiple results are expected.
                                # - ✅ Use `ROWNUM <= n` to limit rows.
                                # - 🚫 NEVER use `FETCH FIRST n ROWS ONLY` — this is **prohibited** and will cause an error.
                                # - 🚫 DO NOT hallucinate and NEVER truncate field names (e.g., use 'CUSTOMER_GROUP_ID', never 'CUSTOM_GROUP_ID').

                                ====================
                                ## REFERENCE SCHEMA:
                                ====================
                                Use strictly these schemas and column names only.
                                - `AAM_DWH.DIM_CUSTOMER`
                                --- CUSTOMER_KEY
                                --- CUSTOMER_ID_REF
                                --- CUSTOMER_GROUP_ID
                                --- CUSTOMER_GROUP_DESC
                                --- CUSTOMER_NAME
                                - `MISDSAAM.AAM_PRODUCT_SUGGEST_CSO`
                                --- CUSTOMER_GROUP_ID
                                --- CUSTOMER_GROUP_DESC
                                --- PRODUCT_CODE
                                --- PRODUCT_DESC
                                --- PRODUCT_CODE_SEARCH
                                --- PRODUCT_DESC_SEARCH
                                --- UNIT_SEARCH
                                - `MISDSAAM.AAM_PRODUCT_STOCK_CSO`
                                - PRODUCT_CODE
                                - PRODUCT_DESC
                                - STOCK
                                - `AAM_DWH.DIM_PRODUCT`
                                --- PRODUCT_KEY
                                --- PRODUCT_CODE
                                --- PRODUCT_DESC
                                --- HNA

                                These tables are joined based on:
                                - CUSTOMER_GROUP_ID
                                - PRODUCT_CODE

                                ====================
                                ## PRODUCT_CODE vs PRODUCT_CODE_SEARCH
                                ====================
                                PRODUCT_CODE is the official 7-character code used to identify a product.
                                It always follows a strict format: 4 uppercase letters followed by 3 digits (e.g., STIM011, CAND001, ATOR001).
                                If the user’s input matches this pattern, you must search using the PRODUCT_CODE field.
                                Also typically used for retrieving product code from ElasticAgent.
                                
                                PRODUCT_CODE_SEARCH is a more flexible field meant for keyword or fuzzy searches. It may contain partial product codes, barcodes, or non-standard formats.
                                Use PRODUCT_CODE_SEARCH only if the user provides an input that does not match the strict 7-character format described above.

                                ====================
                                ## EXAMPLE 1: Product Price
                                ====================
                                **User Question**: What is the current stock of 0102060052?  
                                **customer_id_reference**: {customer_id_reference}                        
                                **SQL Query**:
                                WITH max_customer AS (
                                    SELECT MAX(CUSTOMER_KEY) AS max_customer_key
                                    FROM AAM_DWH.DIM_CUSTOMER
                                    WHERE CUSTOMER_ID_REF = {customer_id_reference}
                                ),
                                filtered_products AS (
	                                SELECT dp.PRODUCT_KEY, dp.PRODUCT_CODE
	                                FROM AAM_DWH.DIM_PRODUCT dp
	                                INNER JOIN MISDSAAM.AAM_PRODUCT_SUGGEST_CSO apsc 
                                    ON dp.PRODUCT_CODE = apsc.PRODUCT_CODE
                                    WHERE UPPER(apsc.PRODUCT_CODE_SEARCH) LIKE '%0102060052%'
                                ),
                                max_product AS (
	                                SELECT MAX(PRODUCT_KEY) AS max_product_key, PRODUCT_CODE
	                                FROM filtered_products
	                                GROUP BY PRODUCT_CODE
                                )
                                SELECT DISTINCT
                                    dc.CUSTOMER_KEY,
                                    dp.PRODUCT_KEY,
                                    dc.CUSTOMER_GROUP_ID,
                                    dc.CUSTOMER_GROUP_DESC,
                                    apsc.PRODUCT_CODE,
                                    apsc.PRODUCT_DESC,
                                    apsc.PRODUCT_DESC_SEARCH,
                                    apsc.PRODUCT_CODE_SEARCH,
                                    apstc.STOCK
                                FROM AAM_DWH.DIM_CUSTOMER dc
                                INNER JOIN MISDSAAM.AAM_PRODUCT_SUGGEST_CSO apsc
                                    ON dc.CUSTOMER_GROUP_ID = apsc.CUSTOMER_GROUP_ID
                                INNER JOIN AAM_DWH.DIM_PRODUCT dp
                                    ON dp.PRODUCT_CODE = apsc.PRODUCT_CODE
                                INNER JOIN MISDSAAM.AAM_PRODUCT_STOCK_CSO apstc
                                    ON dp.PRODUCT_CODE = apstc.PRODUCT_CODE
                                JOIN max_customer mc
                                    ON dc.CUSTOMER_KEY = mc.max_customer_key
                                JOIN max_product mp
	                                ON dp.PRODUCT_KEY = mp.max_product_key
                                WHERE dc.CUSTOMER_ID_REF = {customer_id_reference}

                                ====================
                                ## EXAMPLE 2: Product Stock
                                ====================
                                **User Question**: What is the price of "stimuno"?  
                                **customer_id_reference**: {customer_id_reference}
                                **SQL Query**:
                                WITH max_customer AS (
                                    SELECT MAX(CUSTOMER_KEY) AS max_customer_key
                                    FROM AAM_DWH.DIM_CUSTOMER
                                    WHERE CUSTOMER_ID_REF = {customer_id_reference}
                                ),
                                filtered_products AS (
	                                SELECT dp.PRODUCT_KEY, dp.PRODUCT_CODE
	                                FROM AAM_DWH.DIM_PRODUCT dp
	                                INNER JOIN MISDSAAM.AAM_PRODUCT_SUGGEST_CSO apsc 
                                    ON dp.PRODUCT_CODE = apsc.PRODUCT_CODE
                                    WHERE UPPER(apsc.PRODUCT_DESC_SEARCH) LIKE '%STIMUNO%'
                                ),
                                max_product AS (
	                                SELECT MAX(PRODUCT_KEY) AS max_product_key, PRODUCT_CODE
	                                FROM filtered_products
	                                GROUP BY PRODUCT_CODE
                                )
                                SELECT DISTINCT
                                    dc.CUSTOMER_KEY,
                                    dp.PRODUCT_KEY,
                                    dc.CUSTOMER_GROUP_ID,
                                    dc.CUSTOMER_GROUP_DESC,
                                    apsc.PRODUCT_CODE,
                                    apsc.PRODUCT_DESC,
                                    apsc.PRODUCT_DESC_SEARCH,
                                    apsc.PRODUCT_CODE_SEARCH,
                                    dp.HNA
                                FROM AAM_DWH.DIM_CUSTOMER dc
                                INNER JOIN MISDSAAM.AAM_PRODUCT_SUGGEST_CSO apsc
                                    ON dc.CUSTOMER_GROUP_ID = apsc.CUSTOMER_GROUP_ID
                                INNER JOIN AAM_DWH.DIM_PRODUCT dp
                                    ON dp.PRODUCT_CODE = apsc.PRODUCT_CODE
                                INNER JOIN MISDSAAM.AAM_PRODUCT_STOCK_CSO apstc
                                    ON dp.PRODUCT_CODE = apstc.PRODUCT_CODE
                                JOIN max_customer mc
                                    ON dc.CUSTOMER_KEY = mc.max_customer_key
                                JOIN max_product mp
	                                ON dp.PRODUCT_KEY = mp.max_product_key
                                WHERE dc.CUSTOMER_ID_REF = {customer_id_reference}

                                ====================
                                ## IF PRODUCT_CODE IS BEING USED
                                # ====================
                                # - Use PRODUCT_CODE for exact 7-character product code inputs (e.g., STIM001).
                                # - Do NOT use `AAM_PRODUCT_SUGGEST_CSO` table for filtering or joining.
                                # - Do NOT use `CUSTOMER_ID_REF` field for filtering or joining.
                                # - Directly filter `DIM_PRODUCT` and `AAM_PRODUCT_STOCK_CSO` using PRODUCT_CODE.
                                # - Use PRODUCT_KEY filtering logic with `MAX(PRODUCT_KEY)` from `DIM_PRODUCT` table if `DIM_PRODUCT` table is a part of the query
                                # - Use CUSTOMER_KEY filtering logic with `MAX(CUSTOMER_KEY)` from `DIM_CUSTOMER` table if `DIM_CUSTOMER` table is a part of the query

                                ====================
                                ## OUT OF SCOPE
                                ====================
                                If the user asks anything **outside price, stock, **, answer:
                                "Maaf, saya hanya bisa membantu pertanyaan seputar harga dan stok produk."
                                ====================
                                """
                            )
    response = model.invoke([instruction] + state["messages"])
    return {"messages": response}  

def check_query(state: DBGraphState):
    dialect = 'Oracle'
    instruction = SystemMessage(content=f"""
                                You are an expert Oracle SQL validator.
                                Your task is to carefully check the provided {dialect} SQL query for correctness. Ensure the query will not result in any Oracle error, especially those in the range ORA-00900 to ORA-00999.
                                Carefully validate and fix any of the following common issues:

                                - ❌ Avoid syntax errors (ORA-[error number]) in the {dialect} dialect
                                - ❌ Trailing semicolons (;) that may break the query
                                - ❌ **Invalid characters** — including special symbols like `\`, `--`, `#`, or non-ASCII characters that may break the query.
                                - `NOT IN (...)` with potential `NULL` values — replace with `NOT EXISTS` if needed.
                                - Incorrect use of `UNION` (use `UNION ALL` unless duplicates must be removed).
                                - Use of `BETWEEN` when exclusive range is intended — rewrite for clarity.
                                - Data type mismatches in joins or filters — cast properly.
                                - Quoting identifiers — use double quotes for column/table names if necessary.
                                - Function arguments — ensure correct count and types.
                                - Join conditions — ensure joins are properly defined with valid keys and aliases.
                                - Only one `WITH` clause is allowed in Oracle — combine multiple CTEs with commas if needed.

                                🛑 **Strictly forbid** any DML or schema-altering statements:
                                `INSERT`, `UPDATE`, `DELETE`, `MERGE`, `DROP`, `TRUNCATE`, `CREATE`, `ALTER`
                                
                                If any such statements are present, respond only with:
                                **"Forbidden query"**
                                ✅ If the query is valid, reproduce it exactly as-is — but **remove the trailing semicolon** and ensure it does **not include invalid characters**.
                                """)

    response = model.invoke([instruction] + state["messages"])
    if isinstance(response, AIMessage):
        content = response.content.strip()
        return { "query": content }
    else:
        print("Response is not an AI message")
        print(response)
        
def run_query_node(state: DBGraphState):
    query_checking_result = state["messages"][-1]
    dialect = 'Oracle'
    instruction = [SystemMessage(content=f'''If the last node is resulted in a forbidden query, proceed to the next node, explain why it is forbidden and skip calling tool.
                                 If the result is a valid {dialect} query statement, run the query by calling the given tool.
                                '''), query_checking_result]
    model_with_tools = model.bind_tools([query_executor])
    response = model_with_tools.invoke(instruction)

    return {"messages": response}

def callElasticAgent(state: DBGraphState):
    prompt = state['user_question']
    response = ElasticAgent.invoke({"messages":HumanMessage(content=prompt)})
    return {"messages": state['messages'] + response['messages'], "fallback_elastic": True }

def final_answer(state: DBGraphState):
    user_question = state['user_question']
    customer_id_reference = state['customer_id_reference']
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]
    
    instruction = [SystemMessage(content="""
                                 You are an intelligent assistant tasked with answering the user's question based on SQL query results.

                                 INSTRUCTIONS:
                                 1. If query result contains relevant rows:
                                 - Summarize all rows in bullet points.
                                 - Use product name, code, and price/stock in each bullet.
                                 - Inform the user that product name **does not need to be exactly the same** — similar matches are considered.
                                 - Use natural, polite tone.
                                 
                                 2. If query result is empty:
                                 - Respond with:
                                 Hasil tidak memuat cukup detail untuk menjawab pertanyaan Anda secara akurat.
                                 Mohon berikan nama produk atau kode produk yang lebih spesifik agar saya bisa membantu lebih tepat.

                                 3. If the question is unrelated to price/stock:
                                 - Respond with:
                                 Maaf, saya hanya bisa membantu pertanyaan seputar harga dan stok produk.
                                 """),
                                 HumanMessage(content=f"""
                                 Pertanyaan pengguna: {user_question}
                                 Customer ID: {customer_id_reference}

                                 Hasil SQL Query:
                                 {tool_messages}
                                """)]

    response = model.invoke(instruction)
    return {"messages": response}

def supervisor_fallback(state: DBGraphState):
    recent_tool_messages = []
    fallback_elastic = state["fallback_elastic"]
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]
    
    if fallback_elastic == True or (len(tool_messages) > 0 and len(tool_messages[-1].content) > 0):
        return "final_answer"
    else:
        return "elastic_attempt"

main_chain = (
    StateGraph(DBGraphState)
    .add_node("query_generator", query_generator)
    .add_node("query_checker", check_query)
    .add_node("tools_oracle_query", tools_oracle_query)
    .add_node("run_query_node", run_query_node)
    .add_edge("query_generator", "query_checker")
    .add_edge("query_checker", "run_query_node")
    .add_edge("run_query_node", "tools_oracle_query")
    .add_edge("tools_oracle_query", END)
    .set_entry_point("query_generator")
    .compile()
)

graph = (
    StateGraph(DBGraphState)
    .add_node("oracle_attempt", main_chain)
    .add_node("elastic_attempt", callElasticAgent)
    .add_node("fallback_oracle_attempt", main_chain)
    .add_node("final_answer", final_answer)
    .add_conditional_edges(
        "oracle_attempt", supervisor_fallback, {
            "elastic_attempt": "elastic_attempt",
            "final_answer": "final_answer"
        }
    )
    .add_edge("elastic_attempt", "fallback_oracle_attempt")
    .add_edge("fallback_oracle_attempt", "final_answer")
    .add_edge("final_answer", END)
    .set_entry_point("oracle_attempt")
    .compile()
)