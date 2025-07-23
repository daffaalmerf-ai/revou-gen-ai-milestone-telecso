from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from typing import Any, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage

import os
import cx_Oracle

from dotenv import load_dotenv
load_dotenv(override=True)

model = init_chat_model("gpt-4.1-mini", model_provider= "openai")

class DBGraphState(MessagesState):
    user_question: Annotated[str, "User question that must be answered by querying the database"]
    customer_id_reference: Annotated[int, "User's customer id reference as unique identifier"]

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

@tool("query_executor", parse_docstring=True)
def query_executor(query: str):
    """
        Execute select queries to Oracle database with the provided database configurations.

        Args:
        query = select statement that will be executed
        
        Return:
        Stringified result(s) of the query.
    """
    conn = oracle_set_connection("DEMO-TELECSO", DB_CONFIG_ORACLE)
    cursor = conn.cursor()

    cursor.execute(query)
    query_result = cursor.fetchall()

    data_string = ""
    if len(query_result) == 0: 
        field_names = "No data is returned."
    else:
        field_names = " | ".join([column[0] for column in cursor.description])
        for record in query_result: 
            data_string += " | ".join([str(cell) for cell in record]) + "\n"

    output_string = f"""{field_names}\n{data_string}\n"""
    return output_string

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

                                ====================
                                ## REFERENCE SCHEMA:
                                ====================
                                - `AAM_DWH.DIM_CUSTOMER`
                                - `MISDSAAM.AAM_PRODUCT_SUGGEST_CSO`
                                - `MISDSAAM.AAM_PRODUCT_STOCK_CSO`
                                - `AAM_DWH.DIM_PRODUCT`

                                These tables are joined based on:
                                - CUSTOMER_GROUP_ID
                                - PRODUCT_CODE

                                ====================
                                ## EXAMPLE 1: Product Price
                                ====================
                                **User Question**: What is the current price of "0102060052"?  
                                **customer_id_reference**: {customer_id_reference}                        
                                **SQL Query**:
                                SELECT CUSTOMER_KEY, PRODUCT_KEY, CUSTOMER_GROUP_ID, CUSTOMER_GROUP_DESC, 
                                    PRODUCT_CODE, PRODUCT_DESC, PRODUCT_CODE_SEARCH, PRODUCT_DESC_SEARCH, HNA
                                FROM (
                                    SELECT 
                                        dc.CUSTOMER_KEY,
                                        dp.PRODUCT_KEY,
                                        dc.CUSTOMER_GROUP_ID,
                                        dc.CUSTOMER_GROUP_DESC,
                                        apsc.PRODUCT_CODE,
                                        apsc.PRODUCT_DESC,
                                        apsc.PRODUCT_CODE_SEARCH,
                                        apsc.PRODUCT_DESC_SERACH,
                                        dp.HNA,
                                        ROW_NUMBER() OVER (
                                            ORDER BY dc.CUSTOMER_KEY DESC, dp.PRODUCT_KEY DESC
                                        ) AS rn
                                    FROM AAM_DWH.DIM_CUSTOMER dc
                                    INNER JOIN MISDSAAM.AAM_PRODUCT_SUGGEST_CSO apsc 
                                        ON dc.CUSTOMER_GROUP_ID = apsc.CUSTOMER_GROUP_ID
                                    INNER JOIN AAM_DWH.DIM_PRODUCT dp 
                                        ON dp.PRODUCT_CODE = apsc.PRODUCT_CODE
                                    WHERE dc.CUSTOMER_ID_REF = {customer_id_reference}
                                AND apsc.PRODUCT_CODE_SEARCH = '0102060052'
                                )
                                WHERE rn = 1;

                                ====================
                                ## EXAMPLE 2: Product Stock
                                ====================
                                **User Question**: What is the stock of "stimuno"?  
                                **customer_id_reference**: {customer_id_reference}
                                **SQL Query**:
                                WITH max_customer AS (
                                    SELECT MAX(CUSTOMER_KEY) AS max_customer_key
                                    FROM AAM_DWH.DIM_CUSTOMER
                                    WHERE CUSTOMER_ID_REF = {customer_id_reference}
                                )
                                SELECT 
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
                                WHERE dc.CUSTOMER_ID_REF = {customer_id_reference}
                                AND apsc.PRODUCT_DESC_SEARCH LIKE '%STIM%';

                                ====================
                                ## OUT OF SCOPE
                                ====================
                                If the user asks anything **outside price or stock**, answer:
                                "Maaf, saya hanya bisa membantu pertanyaan seputar harga dan stok produk."
                                ====================
                                """
                            )
    response = model.invoke([instruction] + state["messages"])
    return {"messages": response}  

def check_query(state: DBGraphState):
    dialect = 'Oracle'
    instruction = SystemMessage(content=f'''You are a SQL expert with a strong attention to detail.
    Double check the {dialect} query for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins

    If there are any of the above mistakes, rewrite the query. If there are no mistakes,
    just reproduce the original query.

    Forbid any DML statements (INSERT, UPDATE, DELETE, DROP, TRUNCATE). If the query statement contains those statements, respond by "Forbidden query"
    ''')

    response = model.invoke([instruction] + state["messages"])
    if isinstance(response, AIMessage):
        content = response.content.strip()
        return { "query": content }
    else:
        print("Response is not an AI message")
        print(response)
        
def run_query_node(state: DBGraphState):
    query_checking_result = state["messages"][-1]
    dialect = 'sqlite'
    instruction = [SystemMessage(content=f'''If the last node is resulted in a forbidden query, proceed to the next node, explain why it is forbidden and skip calling tool.
                                 If the result is a valid {dialect} query statement, run the query by calling the given tool.
                                '''), query_checking_result]
    model_with_tools = model.bind_tools([query_executor])
    response = model_with_tools.invoke(instruction)

    return {"messages": response}

def final_answer(state: DBGraphState):
    user_question = state['user_question']
    customer_id_reference = state['customer_id_reference']
    query_result = state['messages'][-1]
    
    instruction = [SystemMessage(content=f'''You are an intelligent assistant tasked with answering the user's question based on query results from a SQL database.
Carefully analyze the SQL result and generate a natural language response that directly answers the user's question.

**INSTRUCTIONS**:
1. Determine if the SQL query result contains enough data to answer the user question.
2. If YES:
   - Extract and show **all rows** of relevant products.
   - Replace placeholders like `[PRODUCT_DESC_SEARCH]`, `[PRODUCT_CODE_SEARCH]`, and `[HNA]` with actual values from the query.
   - Summarize the result clearly using bullet points (one bullet per row).
   - Use a **natural and polite tone**.
3. Replace all placeholders (e.g., [PRODUCT_DESC_SEARCH], [PRODUCT_CODE_SEARCH], [HNA], etc.) with actual values from the SQL result.
4. If multiple results are returned, summarize them clearly using bullet points.
5. If the query does **NOT** provide enough information, politely ask the user for a more specific product name or detail.
   - Example: 
     ```
     Hasil tidak memuat cukup detail untuk menjawab pertanyaan Anda secara akurat.
     Mohon berikan nama produk atau kode produk yang lebih spesifik agar saya bisa membantu lebih tepat.
     ```
6. If the question is **not related to price or stock**, respond with:
  "Maaf, saya hanya bisa membantu pertanyaan seputar harga dan stok produk."

**USER QUESTION**: {user_question}
**CUSTOMER_ID_REFERENCE** {customer_id_reference}

**QUERY RESULT**:
{query_result}

====================
## REFERENCE SCHEMA:
====================
- `AAM_DWH.DIM_CUSTOMER`
- `MISDSAAM.AAM_PRODUCT_SUGGEST_CSO`
- `MISDSAAM.AAM_PRODUCT_STOCK_CSO`
- `AAM_DWH.DIM_PRODUCT`

These tables are joined based on:
- CUSTOMER_GROUP_ID
- PRODUCT_CODE

====================
## EXAMPLE 1: Product Price
====================
**User Question**: Berapa harga "0102060052"?  
**Answer**:  
Harga dari produk [PRODUCT_DESC_SEARCH] dengan kode [PRODUCT_CODE_SEARCH], yaitu [PRODUCT_DESC] adalah Rp. [HNA].

====================
## EXAMPLE 2: Product Stock
====================
**User Question**: Ada stok untuk "stimuno"?  
**Answer**:  
Stok dari produk [PRODUCT_DESC_SEARCH] dengan kode [PRODUCT_CODE_SEARCH], yaitu [PRODUCT_DESC] adalah [STOCK].

====================
## OUT OF SCOPE
====================
If the user asks anything outside price or stock:
"Maaf, saya hanya bisa membantu pertanyaan seputar harga dan stok produk."
''')]

    response = model.invoke(instruction)
    return {"messages": response}

graph = (
    StateGraph(DBGraphState)
    .add_node("query_generator", query_generator)
    .add_node("query_checker", check_query)
    .add_node("tools_oracle_query", tools_oracle_query)
    .add_node("run_query_node", run_query_node)
    .add_node("final_answer", final_answer)
    .add_edge("query_generator", "tools_oracle_query")
    .add_edge("tools_oracle_query", "query_checker")
    .add_edge("query_checker", "run_query_node")
    .add_edge("run_query_node", "final_answer")
    .add_edge("final_answer", END)
    .set_entry_point("query_generator")
    .compile()
)