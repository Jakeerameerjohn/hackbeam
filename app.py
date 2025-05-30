import base64
import os
import streamlit as st
from datetime import datetime
from backend.knowledge_base import initialize_collection
from backend.rag_engine import get_rag_answer
from backend.config import EMBEDDING_MODEL_NAME, HF_LLM_MODEL_ID, HF_API_TOKEN, HF_API_URL, KB_FILE_PATH

# ---------- Title ----------
st.title("💡 Lightcast KB Chatbot")
st.markdown("Ask questions and get answers from the Lightcast documentation.")
logo_path = os.path.join("assets", "lightcast-logo.png")

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image(logo_path)

st.sidebar.markdown(
    f"""
    <div style='display: flex; align-items: center; gap: 10px; padding-bottom: 15px;'>
        <img src='data:image/png;base64,{logo_base64}' width='40' style='border-radius: 5px;' />
        <span style='font-size: 22px; font-weight: bold; color: #DC143C;'>RAGnarok</span>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- Initialize Chroma Collection ----------
collection = initialize_collection()

# ---------- State Management ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = {}
if "current_chat_id" not in st.session_state:
    chat_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.current_chat_id = chat_id
    st.session_state.history[chat_id] = []


# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
if prompt := st.chat_input("Ask a question about Lightcast documentation..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating response..."):
            full_answer_text, sources = get_rag_answer(prompt, collection, HF_API_URL, HF_API_TOKEN,
                                                       st.session_state.messages)
            st.markdown(full_answer_text)
            st.session_state.messages.append({"role": "assistant", "content": full_answer_text})

    if not os.path.isdir(KB_FILE_PATH):
        st.error(
            f"Error: Knowledge base directory '{KB_FILE_PATH}' not found. Please create it and add your documentation files.")
        st.stop()

# ---------- Sidebar ----------
st.sidebar.markdown(
    "<h3 style='margin-top: 10px;'>💬 Chat History</h3>",
    unsafe_allow_html=True
)
if st.sidebar.button("➕ New Chat"):
    chat_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    st.session_state.history[chat_id] = []
    st.rerun()

# Chat selector
selected_chat = st.sidebar.radio("Previous Chats", list(st.session_state.history.keys())[::-1], key="chat_select")

if selected_chat != st.session_state.current_chat_id:
    st.session_state.current_chat_id = selected_chat
    st.session_state.messages = st.session_state.history[selected_chat]
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown(f"**Knowledge Base Directory:** `{KB_FILE_PATH}`")
st.sidebar.markdown(f"**Embedding Model:** `{EMBEDDING_MODEL_NAME}`")
st.sidebar.markdown(f"**LLM Model:** `{HF_LLM_MODEL_ID}` (Hugging Face Inference API)")
with st.sidebar.expander("About This App"):
    st.write(
        "RAGnarok is an AI chatbot that provides instant, conversational answers from "
        "a subset of the Lightcast documentation. It uses Retrieval-Augmented Generation "
        "to extract relevant sections and generate accurate, context-aware responses "
        "with source links."
    )
    st.info("Version 1.0")
st.sidebar.markdown("---")
st.sidebar.caption("Hackathon Prototype - May 2025")
