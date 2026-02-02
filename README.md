<<<<<<< HEAD
# 📄 Multi-Utility LangGraph PDF Chatbot

A **Streamlit-based conversational AI application** powered by **LangGraph**, **Groq LLM**, and **FAISS**, supporting **PDF-based RAG**, **tool usage**, and **multi-threaded persistent chat history**.

---

## 🚀 Features

### 🔹 Conversational AI
- Uses **Groq-hosted LLM (`openai/gpt-oss-20b`)**
- Maintains **conversation memory** per chat thread
- Streams responses token-by-token for real-time UX

### 🔹 PDF Question Answering (RAG)
- Upload a PDF per chat thread
- Documents are:
  - Loaded with `PyPDFLoader`
  - Chunked using `RecursiveCharacterTextSplitter`
  - Embedded via **HuggingFace MiniLM**
  - Stored in **FAISS**
- Context is retrieved **only when required** using a dedicated `rag_tool`

### 🔹 Built-in Tools
The assistant can autonomously decide to use:
- 🔍 DuckDuckGo Search (current information)
- 🧮 Calculator (add, subtract, multiply, divide)
- 🌦 Weather lookup
- 📈 Stock price lookup (Alpha Vantage)
- ⏰ Current date & time
- 📄 PDF RAG tool (thread-aware)

> Tool outputs are **never shown directly** to the user — only final assistant responses.

---

## 🧠 Architecture Overview

```
User (Streamlit UI)
   ↓
LangGraph StateGraph
   ├── chat_node (LLM reasoning)
   ├── ToolNode (tools execution)
   └── SQLite Checkpointer (state persistence)
```

PDF Flow:
```
PDF → PyPDFLoader → Chunking → Embeddings → FAISS → rag_tool
```

---

## 🗂 Project Structure

```
.
├── backend_rag_chatbot.py   # LangGraph backend logic
├── app.py                  # Streamlit frontend
├── chatbot.db              # SQLite checkpoint database
├── .env                    # API keys
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

### Backend
- LangGraph
- LangChain
- Groq LLM
- FAISS
- HuggingFace Embeddings
- SQLite (checkpointing)

### Frontend
- Streamlit
- Streaming chat UI
- Session-based thread management

---

## 🔐 Environment Variables

Create a `.env` file:

```env
groq_api=YOUR_GROQ_API_KEY
WEATHERSTACK_API_KEY=YOUR_WEATHERSTACK_KEY
ALPHAVANTAGE_API_KEY=YOUR_ALPHA_VANTAGE_KEY
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

---

## 🧪 How It Works

1. User starts or selects a chat thread
2. (Optional) Uploads a PDF
3. User sends a message
4. LangGraph decides whether to call a tool
5. RAG is used **only** for PDF-related questions
6. Responses stream live to the UI
7. State and metadata persist in SQLite

---

## 🧵 Thread Persistence

- Each thread stores:
  - Messages
  - Title
  - Creation timestamp
- Switching threads restores the full conversation
- PDFs are isolated **per thread**

---

## 🛡 Design Constraints

- Assistant must **either respond OR call one tool**
- Tool outputs are internal only
- RAG is strictly limited to uploaded PDFs
- Context window capped to avoid overflow

---

## 🏁 Summary

This project demonstrates a **production-grade LangGraph chatbot** with:
- Multi-threaded memory
- Tool-augmented reasoning
- Thread-aware RAG
- Streaming UI
- Persistent state

Ideal for real-world conversational AI systems — not just demos.

=======
# multi-tool-rag-chatbot
>>>>>>> eb6031ffed5bcfe5935e8344f32c5aa626e5633a
