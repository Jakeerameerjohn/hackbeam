import os
import glob
from chromadb import Client
from backend.embeddings import MyEmbeddingFunction
from backend.config import KB_FILE_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME
import streamlit as st


def load_documents(file_path=KB_FILE_PATH):
    documents = []

    # If file_path is a directory, iterate over all .txt files inside it
    if os.path.isdir(file_path):
        txt_files = glob.glob(os.path.join(file_path, "*.txt"))
    else:
        txt_files = [file_path]

    for file_index, txt_file in enumerate(txt_files):
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
            parts = content.split('---')
            for i, part in enumerate(parts):
                if part.strip():
                    lines = part.strip().split('\n')
                    title = "Untitled Document"
                    source = "Unknown"
                    content_text = ""

                    if lines[0].lower().startswith("title:"):
                        title = lines[0].split(":", 1)[1].strip()
                        if len(lines) > 1 and lines[1].lower().startswith("source:"):
                            source = lines[1].split(":", 1)[1].strip()
                            content_text = "\n".join(lines[2:]).replace("Content:", "", 1).strip()
                        else:
                            content_text = "\n".join(lines[1:]).replace("Content:", "", 1).strip()
                    else:
                        content_text = "\n".join(lines).strip()

                    if content_text:
                        documents.append({
                            "id": f"doc_{file_index}_{i}",
                            "title": title,
                            "source": source,
                            "content": content_text
                        })
    return documents

def initialize_collection():
    client = Client()
    embedding_function = MyEmbeddingFunction(EMBEDDING_MODEL_NAME)
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)

    if collection.count() == 0:
        documents = load_documents()
        st.info(f"Populating knowledge base with {len(documents)} documents. This may take a moment...")
        if not documents:
            st.warning(f"No documents loaded from '{KB_FILE_PATH}'. Please ensure it contains correctly formatted files.")
            st.stop()
        ids = [doc["id"] for doc in documents]
        contents = [doc["content"] for doc in documents]
        metadatas = [{"title": doc["title"], "source": doc["source"]} for doc in documents]
        collection.add(documents=contents, metadatas=metadatas, ids=ids)
    
    return collection
