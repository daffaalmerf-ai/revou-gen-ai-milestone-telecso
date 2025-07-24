import streamlit_authenticator as stauth
import streamlit as st
import yaml
from yaml.loader import SafeLoader
import os
from langchain_core.messages import AIMessageChunk, HumanMessage, AIMessage, SystemMessage
import agents.oracle_cache_agent as OracleAgent
import agents.elasticsearch as ElasticAgent
import agents.milvus as MilvusAgent
import agents.ocr as OcrAgent
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command
from typing import Literal
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
import json
import base64
from guardrails.hub import RestrictToTopic
from guardrails import Guard

from dotenv import load_dotenv
load_dotenv(override=True)

DB_CONFIG_ORACLE = {
    "HOST": os.environ["ORACLE_HOST"],
    "PORT": os.environ["ORACLE_PORT"],
    "SERVICE_NAME": os.environ["ORACLE_SERVICE_NAME"],
    "USERNAME": os.environ["ORACLE_USERNAME"],
    "PASSWORD": os.environ["ORACLE_PASSWORD"]
}

if "customer_id_ref_mapping" not in st.session_state:
    st.session_state["customer_id_ref_mapping"] = {
        "40 PRIMAYA, RS": 193763,
        "34 EKA HOSPITAL BEKASI": 186010,
        "43 GAMA / A. YANI, APT": 174490,
        "30 PONDOK INDAH, RS.": 126851
    }

if "customer_id_ref" not in st.session_state:
    st.session_state["customer_id_ref"] = None
if "chat_memory" not in st.session_state:
    st.session_state["chat_memory"] = dict()
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = dict()

model = init_chat_model("gpt-4.1-mini", model_provider= "openai")

with open('./src/config/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

authenticator.login('main')

if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None
if "name" not in st.session_state:
    st.session_state["name"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None

class BestAgent(BaseModel):
    agent_name: str = Field(description = "The best agent to handle specific request from users.")

class SupervisorState(MessagesState):
    user_question: str
    customer_id_ref: int
    oracle_found: bool

guard = Guard().use(
    RestrictToTopic(
        valid_topics=["products", "orders", "invoices", "stock"],
        invalid_topics=[
            "privacy", "other users", "other customers' purchases",
            "crime", "criminal activity", "theft", "fraud",
            "drug misuse", "illegal drugs", "narcotics", "drug abuse", "misuse of drugs"
        ],
        disable_classifier=True,
        disable_llm=False,
        on_fail="exception"
    )
)

def supervisor(state: SupervisorState) -> Command[Literal["OracleAgent", "ElasticAgent", "MilvusAgent", END]]:
    last_message = state["messages"][-1]
    customer_id_ref = state["customer_id_ref"]
    instruction = [SystemMessage(content=f"""
        You are an intelligent assistant tasked with routing user queries to the appropriate agent. Based on the user's last message and the previous agent's response, determine the next action.
        Rules:
        - If the user's question is related to **product recommendations** based on similarity or product details strictly related to description, indication, usage, how to use, dosage, and side effects, delegate to the **MilvusAgent**.
        - If the user's question is about **price or stock availability**, delegate to the **OracleAgent**.
        - If the **OracleAgent** fails to return the desired product, use **ElasticAgent** as a fallback to try and fetch the appropriate product.
        - If a sufficient and complete answer has already been provided, you may choose to **END** the conversation.
        - Ensure to always prioritize user clarity, provide concise, relevant information, and maintain a friendly, professional tone.
    
        You must reply ONLY with a JSON object like this:
        {{ "agent_name": "OracleAgent" }}
        """)
    ]
    model_with_structure = model.with_structured_output(BestAgent)
    response = model_with_structure.invoke(instruction + [last_message])
    return Command(
        update= {'user_question': last_message.content, 'customer_id_ref': customer_id_ref},
        goto=response.agent_name
    )

def callOracleAgent(state: SupervisorState) -> Command[Literal['supervisor']]:
    try:
        prompt = state['user_question']
        customer_id_ref = state['customer_id_ref']
        response = OracleAgent.graph.invoke({"messages":[HumanMessage(content=json.dumps({
            "user_question": prompt,
            "db_config": DB_CONFIG_ORACLE,
            "customer_id_reference": customer_id_ref,
        }))], "db_config": DB_CONFIG_ORACLE, "user_question" : prompt, "customer_id_reference": customer_id_ref, "fallback_elastic": False})
        return Command(
            goto=END,
            update={"messages": response['messages'][-1]}
        )
    except Exception as e:
        return Command(
            update={"messages": AIMessage(content="Mohon maaf, pertanyaan Anda di luar cakupan layanan kami. Silakan ajukan pertanyaan terkait produk, pesanan, faktur, atau stok, dan terbatas pada riwayat pembelian Anda sendiri. Kami tidak dapat menanggapi pertanyaan seputar privasi pelanggan lain, isu kriminal, penyalahgunaan obat, atau topik yang tidak relevan dengan layanan kami.")},
            goto=END
        )

def callElasticAgent(state: SupervisorState) -> Command[Literal['supervisor']]:
    try:
        prompt = state['user_question']
        response = ElasticAgent.graph.invoke({"messages":HumanMessage(content=prompt)})
        return Command(
            goto=END,
            update={"messages": response['messages'][-1]}
        )
    except Exception as e:
        return Command(
            update={"messages": AIMessage(content="Mohon maaf, pertanyaan Anda di luar cakupan layanan kami. Silakan ajukan pertanyaan terkait produk, pesanan, faktur, atau stok, dan terbatas pada riwayat pembelian Anda sendiri. Kami tidak dapat menanggapi pertanyaan seputar privasi pelanggan lain, isu kriminal, penyalahgunaan obat, atau topik yang tidak relevan dengan layanan kami.")},
            goto=END
        )

def callMilvusAgent(state: SupervisorState) -> Command[Literal['supervisor']]:
    try:
        prompt = state['user_question']
        response = MilvusAgent.graph.invoke({"messages":HumanMessage(content=prompt)})
        return Command(
            goto=END,
            update={"messages": response['messages'][-1]}
        )
    except Exception as e:
        return Command(
            update={"messages": AIMessage(content="Mohon maaf, pertanyaan Anda di luar cakupan layanan kami. Silakan ajukan pertanyaan terkait produk, pesanan, faktur, atau stok, dan terbatas pada riwayat pembelian Anda sendiri. Kami tidak dapat menanggapi pertanyaan seputar privasi pelanggan lain, isu kriminal, penyalahgunaan obat, atau topik yang tidak relevan dengan layanan kami.")},
            goto=END
        )

def build_agent():
    memory = InMemorySaver()
    supervisor_agent = (
        StateGraph(SupervisorState)
        .add_node(supervisor)
        .add_node("OracleAgent", callOracleAgent)
        .add_node("ElasticAgent", callElasticAgent)
        .add_node("MilvusAgent", callMilvusAgent)
        .add_edge(START, "supervisor")
        .compile(name= "supervisor", checkpointer=memory)
    )
    return memory, supervisor_agent

memory = None
supervisor_agent = None

if st.session_state["authentication_status"]:
    authenticator.logout('Logout')
    st.title("AAM Customer Service (TeleCSO)")
    st.write(f'Welcome *{st.session_state["name"]}*')
    if st.session_state["name"] in st.session_state["customer_id_ref_mapping"]:
        customer_id_ref = st.session_state["customer_id_ref_mapping"][st.session_state["name"]]
        st.session_state["customer_id_ref"] = customer_id_ref
        if customer_id_ref not in st.session_state["chat_history"].keys():
            st.session_state["chat_history"][customer_id_ref] = []
        if customer_id_ref not in st.session_state["chat_memory"].keys():
            st.session_state["chat_memory"][customer_id_ref] = dict()
            memory, supervisor_agent = build_agent()
            st.session_state["chat_memory"][customer_id_ref]["memory"] = memory
            st.session_state["chat_memory"][customer_id_ref]["agent"] = supervisor_agent
        else:
            memory = st.session_state["chat_memory"][st.session_state["customer_id_ref"]]["memory"]
            supervisor_agent = st.session_state["chat_memory"][st.session_state["customer_id_ref"]]["agent"]

    config = {"configurable": {"thread_id": st.session_state['customer_id_ref']}}

    curr_chat_history = st.session_state["chat_history"][st.session_state['customer_id_ref']]

    if curr_chat_history:
        for chat in curr_chat_history:
            if chat["file"]:
                st.image(chat["file"])
            with st.chat_message("human"):
                st.markdown(chat["question"])
            with st.chat_message("ai"):
                st.markdown(chat["answer"])

    prompt = st.chat_input("Write your question here ... ", accept_file=True,
                           file_type=["png", "jpg", "jpeg"],)
    if prompt and prompt.text:
        question = ""
        answer = ""
        
        uploaded_file = None
        filename = ""
        image_base64 = None

        if prompt and prompt["files"]:
            uploaded_file = prompt["files"][0]
            st.image(prompt["files"][0])
            file_bytes = uploaded_file.read()
            filename = uploaded_file.name
            image_base64 = base64.b64encode(file_bytes).decode("utf-8")

        if prompt.text:
            with st.chat_message("human"):
                st.markdown(prompt.text)
                question = prompt.text

        final_answer = ""

        with st.chat_message("ai"):
            status_placeholder = st.empty()
            question_placeholder = st.empty()
            answer_placeholder = st.empty()
            status_placeholder.status(label="Process Start")
            state = "Process Start"

            prior_state = st.session_state["chat_memory"][customer_id_ref]["memory"]
            # for prior_message in st.session_state["chat_history"][customer_id_ref]:
            #     prior_messages.append(HumanMessage(content=prior_message["question"]))
            #     prior_messages.append(AIMessage(content=prior_message["answer"]))
            
            # new_input = {
            #     "messages": prior_messages + [HumanMessage(content=prompt.text)],
            #     "customer_id_ref": customer_id_ref
            # }
            
            if image_base64:
                response = OcrAgent.graph.invoke({
                    "messages": [HumanMessage(content=json.dumps({
                        "image_base64": image_base64,
                        "filename": filename
                    }))]
                })
                final_answer = response["messages"][-1].content
                answer_placeholder.markdown(final_answer)
            else:
        
                for chunk, metadata in supervisor_agent.stream({"messages": [HumanMessage(content=prompt.text)], "customer_id_ref": customer_id_ref}, stream_mode="messages", config=config):
                    if metadata['langgraph_node'] != state:
                        status_placeholder.status(label=metadata['langgraph_node'])
                        state = metadata['langgraph_node']
                        final_answer = ""
                    if metadata['langgraph_node'] == "generate_elastic_code" or metadata['langgraph_node'] == 'generate_similar_product' or metadata['langgraph_node'] == 'final_answer' or metadata['langgraph_node'] == 'query_or_respond_similar_product':
                        final_answer += chunk.content
                        answer_placeholder.markdown(final_answer)
        
        answer = final_answer
        status_placeholder.status(label="Complete", state='complete')
        
        curr_chat_history.append({ "question": question, "answer": answer, "file": uploaded_file if uploaded_file else None })
        st.rerun()

elif st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')