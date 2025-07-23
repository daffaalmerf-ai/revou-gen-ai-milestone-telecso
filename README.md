# Telecso

**Telecso** is a modular and intelligent platform designed to streamline customer support, sales operations, and order management through automation and smart agentic workflows.

---

## 🚀 Features

- 🔍 **Product Discovery**  
  Search for products using product name, code, or semantic similarity.

- 🧠 **Agentic AI System**  
  Utilizes multi-agent orchestration (LangGraph + LangChain) for handling complex tasks:
  - Product lookup
  - Stock checks
  - Smart product recommendations
  - Purchase order processing
  - Invoice generation and order tracking

- 📝 **File-based Order Handling**  
  Upload PDFs or images of purchase orders. AI extracts structured order data.

- 🧾 **Dynamic Invoice Generation**  
  Auto-generates unique invoice numbers based on customer ID and timestamps.

- 📦 **Order Status Tracker**  
  Retrieve real-time order statuses by invoice number.

- 🔒 **User Context Awareness**  
  Filter data and operations based on logged-in customer identity.

---

## 🧰 Tech Stack

- **Backend:** FastAPI, Oracle, LangGraph
- **Frontend:** Streamlit
- **AI & NLP:** GPT-4, Milvus (vector similarity), Elasticsearch (fuzzy search)
- **Embeddings:** `text-embedding-3-small` (OpenAI)
- **Tooling:** LangChain, Guardrails, LangSmith