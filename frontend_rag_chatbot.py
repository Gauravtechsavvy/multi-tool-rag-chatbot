import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from datetime import datetime
from backend_rag_chatbot import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
    generate_title_from_prompt
)

# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

    #  CREATE BACKEND METADATA IMMEDIATELY
    chatbot.update_state(
        config={"configurable": {"thread_id": thread_id}},
        values={
            "thread_meta": {
                "title": "New chat",
                "created_at": datetime.now().strftime("%d %b %H:%M"),
                "has_title": False,
            }
        }
    )


def add_thread(thread_id):
    thread_id = str(thread_id)
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].insert(0, thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

# ======================= Session Initialization ===================

if "selected_thread" not in st.session_state:
    st.session_state['selected_thread'] = None

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if "pdf_processing" not in st.session_state:
    st.session_state["pdf_processing"] = set()

if "last_uploaded_file" not in st.session_state:
    st.session_state["last_uploaded_file"] = {}

# Generate thread_id if missing
if "thread_id" not in st.session_state:
    tid = generate_thread_id()
    st.session_state["thread_id"] = tid
    add_thread(tid)

    #  CREATE THREAD METADATA IMMEDIATELY
    chatbot.update_state(
        config={"configurable": {"thread_id": tid}},
        values={
            "thread_meta": {
                "title": "New chat",
                "created_at": datetime.now().strftime("%d %b %H:%M"),
                "has_title": False,
            }
        }
    )


# Current thread
thread_key = st.session_state["thread_id"]

# Initialize sidebar metadata if missing
if thread_key not in st.session_state["thread_titles"]:
    st.session_state["thread_titles"][thread_key] = {
        "title": generate_title_from_prompt(None),
        "created_at": datetime.now().strftime("%d %b %H:%M"),
        "has_title": False,
    }

# Ensure ingested docs dict exists
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

# Reverse threads for sidebar (recent first)
threads = list(reversed(st.session_state["chat_threads"]))  # no meaning of reversing it unordered by retrieved_all_thread
selected_thread = None

# ============================ Sidebar ============================
st.sidebar.title("LangGraph PDF Chatbot")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1] # at present we are not storing permanently so [-1] not needed
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")


uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat",
    type=["pdf"],
    key=f"uploader-{thread_key}"
)
if uploaded_pdf is not None:
    filename = uploaded_pdf.name

    last_file = st.session_state["last_uploaded_file"].get(thread_key)

    if filename in thread_docs and last_file != filename:
        st.sidebar.info(f"`{filename}` already processed for this chat.")

    elif filename not in thread_docs:
        with st.sidebar.status("Indexing PDF…", expanded=True):
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=filename,
            )
            st.session_state["ingested_docs"].setdefault(thread_key, {})[filename] = summary
            st.session_state["last_uploaded_file"][thread_key] = filename

        st.rerun()



st.sidebar.subheader("Past conversations")

# Collect threads with metadata
thread_items = []

for tid in st.session_state["chat_threads"]:
    state = chatbot.get_state(
        config={"configurable": {"thread_id": tid}}
    )
    meta = state.values.get("thread_meta", {})

    created_at = meta.get("created_at")
    if created_at:
        created_at_dt = datetime.strptime(created_at, "%d %b %H:%M")
    else:
        created_at_dt = datetime.min

    thread_items.append((tid, meta, created_at_dt))

#  Sort: recent first
thread_items.sort(key=lambda x: x[2], reverse=True)

# Render buttons
for i, (tid, meta, _) in enumerate(thread_items):
    title = meta.get("title", "Previous chat")
    created = meta.get("created_at", "")
    label = f"{title} · {created}"

    if st.sidebar.button(label, key=f"side-thread-{tid}-{i}"):
        selected_thread = tid #But Streamlit + reruns + state restoration create situations where the same tid can appear more than once in the sidebar rendering loop, even if the ID itself is unique in backend.


# ============================ Main Layout ========================
st.title("Multi Utility Chatbot")
    
# ============================ Render Chat ========================
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"] or "")

# ============================ User Input =========================
user_input = st.chat_input("Ask about your document or use tools")

if user_input is not None and user_input.strip() != "":
    prompt = user_input.strip()  #  HARD GUARANTEE STRING

    with st.chat_message("user"):
        st.markdown(prompt)

    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_key}}
    )

    thread_meta = state.values.get("thread_meta", {})

    # ---- generate ChatGPT-style title ONCE ----
    if not thread_meta.get("has_title"):
        thread_meta["title"] = generate_title_from_prompt(user_input)
        thread_meta["has_title"] = True

        #  frontend cache
        st.session_state["thread_titles"][thread_key] = thread_meta

        #  persist to backend (SQLite)
        chatbot.update_state(
            config={"configurable": {"thread_id": thread_key}},
            values={"thread_meta": thread_meta}
        )

    # ---- LangGraph config ----
    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    status_holder = {"box": None}
    collected_chunks = []  #  SINGLE SOURCE OF TRUTH

    # ===================== Streaming Generator ====================
    def ai_only_stream():
        for message_chunk, _ in chatbot.stream(
            {"messages": [HumanMessage(content=prompt)]},
            config=CONFIG,
            stream_mode="messages",
        ):
            if isinstance(message_chunk, ToolMessage):
                tool_name = getattr(message_chunk, "name", "tool")
                if status_holder["box"] is None:
                    status_holder["box"] = st.status(
                        f"🔧 Using `{tool_name}` …",
                        expanded=True
                    )
                continue

            if isinstance(message_chunk, AIMessage):
                if message_chunk.content:
                    collected_chunks.append(message_chunk.content)
                    yield message_chunk.content

        # Prevent Streamlit returning None
        if not collected_chunks:
            yield ""

    # ===================== Assistant UI ===========================
    with st.chat_message("assistant"):
        st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False
            )

        # ---- show PDF metadata ----
        doc_meta = thread_document_metadata(thread_key)
        if doc_meta:
            st.caption(
                f"Document indexed: {doc_meta.get('filename')} "
                f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
            )

    # ===================== Final Assistant Text ===================
    full_ai_message = "".join(collected_chunks).strip()

        # ---- Persist BOTH messages to LangGraph ----
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_key}})
    st.session_state["message_history"] = [
        {"role": "user", "content": m.content} if isinstance(m, HumanMessage)
        else {"role": "assistant", "content": m.content}
        for m in state.values.get("messages", [])
        if isinstance(m, (HumanMessage, AIMessage)) and m.content
]

# ============================ Thread Switch ======================
st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread

    state = chatbot.get_state(
        config={"configurable": {"thread_id": selected_thread}}
    )

    temp_messages = []
    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage) and msg.content:
            temp_messages.append(
                {"role": "user", "content": msg.content}
            )
        elif isinstance(msg, AIMessage) and msg.content:
            temp_messages.append(
                {"role": "assistant", "content": msg.content}
            )

    st.session_state["message_history"] = temp_messages
    st.rerun()
