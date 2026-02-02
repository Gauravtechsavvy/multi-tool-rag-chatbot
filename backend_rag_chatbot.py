from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict
from datetime import datetime
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------
llm = ChatGroq(groq_api_key = os.getenv('groq_api'),
                 model ="openai/gpt-oss-20b"
                )
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def generate_title_from_prompt(prompt: str | None) -> str:
    if not prompt:
        return "New chat"

    words = prompt.strip().split()
    return " ".join(words[:5])

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict: #bytes means binary data
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
duckduckgo_tool = DuckDuckGoSearchRun(
    name="duckduckgo_search",
    description="Search the web for current information"
)


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}

@tool
def get_weather_update(city:str) ->str:
    """This tool fetches the current weather updates for the given city"""
    url = f'https://api.weatherstack.com/current?access_key=WEATHERSTACK_API_KEY&query={city}'
    response = requests.get(url)
    return response.json()
@tool
def get_current_datetime() -> str:
    """Get current date and time"""
    return datetime.now().isoformat()

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=ALPHAVANTAGE_API_KEY"
    )
    r = requests.get(url)
    return r.json()


@tool
def rag_tool(query: str | None = None, thread_id: str | None = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for the current chat thread.
    Use this ONLY when the user asks questions about the uploaded document.
    """
    if not query or not thread_id:
        return {"error": "INVALID_ARGUMENTS"}

    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {"error": "NO_DOCUMENT"}

    result = retriever.invoke(query)

    context_text = "\n\n".join(
        doc.page_content[:500] for doc in result[:3]
    )

    return {
        "context": context_text,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }



tools = [duckduckgo_tool, get_stock_price, calculator, rag_tool, get_weather_update, get_current_datetime]

llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict,total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_meta:dict

# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    thread_meta = state.get("thread_meta") or {
        "title": "New chat",
        "created_at": datetime.now().strftime("%d %b %H:%M"),
        "has_title": False,
    }#The fallback code inside chat_node exists for API / backend-only usage, not for your Streamlit UI flow

    system_message = SystemMessage(
        content=(
            "You are a strict tool-using AI assistant.\n\n"
            "CRITICAL RULES:\n"
            "1. You MUST either:\n"
            "   - Respond with normal text, OR\n"
            "   - Call exactly ONE tool\n"
            "   Never do both.\n\n"
            "2. Tool outputs are INTERNAL. Never show them.\n\n"
            "- Use rag_tool ONLY for uploaded PDFs.\n"
            "- rag_tool REQUIRES query + thread_id.\n\n"
            f"Current thread_id: {thread_id}"
        )
    )
    max_messages = 10
    history = state.get("messages", [])
    messages = [system_message, *history[-max_messages:]]# to prevent context window overflow, to protect from error
    response = llm_with_tools.invoke(messages, config=config)

    return {
        "messages": [response],
        "thread_meta": thread_meta,
    }

tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})