import os
import requests
import json
from backend.config import HF_API_URL, HF_API_TOKEN
import streamlit as st

def _rewrite_query(current_query, chat_history, hf_api_url, hf_api_token):
    """
    Rewrites the current query using LLM based on recent chat history to make it standalone.
    """
    if not chat_history or len(chat_history) < 2:  # Need at least one user and one assistant message
        return current_query  # No history to rewrite with

    # Take the last user query and the last assistant response
    last_user_message = next((msg['content'] for msg in reversed(chat_history) if msg['role'] == 'user'), None)
    last_assistant_message = next((msg['content'] for msg in reversed(chat_history) if msg['role'] == 'assistant'),
                                  None)

    if not last_user_message or not last_assistant_message:
        return current_query  # Not enough history to rewrite

    # Clean up assistant's message from sources for better rewriting
    clean_assistant_message = last_assistant_message.split('**Sources:**')[0].strip()

    rewrite_prompt = f"""Given the following conversation history and a follow-up question, rewrite the follow-up question to be a standalone question.
The rewritten question should be clear and understandable without needing the previous context.

Conversation History:
User: {last_user_message}
Assistant: {clean_assistant_message}

Follow-up Question: {current_query}

Rewritten Standalone Question:
"""
    headers = {
        "Authorization": f"Bearer {hf_api_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": rewrite_prompt,
        "parameters": {
            "max_new_tokens": 100,  # Rewritten query should be concise
            "temperature": 0.1,  # Keep it deterministic
            "do_sample": False,
            "return_full_text": False
        },
        "options": {
            "use_cache": False,
            "wait_for_model": True
        }
    }

    try:
        response = requests.post(hf_api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        rewritten_query = result[0]['generated_text'].strip()

        # Simple post-processing to clean up potential unwanted phrases
        if rewritten_query.lower().startswith("rewritten standalone question:"):
            rewritten_query = rewritten_query[len("rewritten standalone question:"):].strip()

        # st.sidebar.text_area("Rewritten Query for Retrieval", rewritten_query)

        return rewritten_query
    except requests.exceptions.RequestException as e:
        st.warning(f"Failed to rewrite query with LLM: {e}. Using original query.")
        return current_query
    except Exception as e:
        st.warning(f"An unexpected error occurred during query rewriting: {e}. Using original query.")
        return current_query


def get_rag_answer(query, collection, hf_api_url, hf_api_token, chat_history, n_results=1):
    """
    Retrieves relevant documents based on a potentially rewritten query and generates an answer using RAG.
    """
    effective_query = query
    # Only rewrite query if there's sufficient history to do so
    if len(chat_history) >= 2:  # At least one full user-assistant turn prior
        effective_query = _rewrite_query(query, chat_history, hf_api_url, hf_api_token)
        # st.sidebar.text(f"Effective Query: {effective_query}")

    # 1. Retrieve relevant documents from ChromaDB using the effective_query
    try:
        results = collection.query(
            query_texts=[effective_query],  # Use the rewritten query here!
            n_results=n_results,
            include=['documents', 'metadatas']
        )
    except Exception as e:
        return f"Error retrieving documents from knowledge base: {e}", []

    retrieved_docs = results['documents'][0]
    retrieved_metadatas = results['metadatas'][0]

    if not retrieved_docs:
        return "No relevant information found in the Lightcast Knowledge Base for your query. Please try rephrasing.", []

    context_parts = []
    source_links = set()
    # sources = set()
    for i, doc in enumerate(retrieved_docs):
        metadata = retrieved_metadatas[i]
        source_url = metadata.get('source', 'N/A')

        if source_url.startswith('file://'):
            display_source = f"Local File: {os.path.basename(source_url.replace('file://', ''))}"
            source_for_link = ""
        else:
            display_source = source_url
            source_for_link = source_url
            if not source_url.startswith('http://') and not source_url.startswith('https://'):
                source_for_link = f"https://{source_url}"

        context_parts.append(
            f"### Document Title: {metadata.get('title', 'N/A')}\n"
            f"Content: {doc}\n"
            f"Source: {display_source}"
        )
        if source_for_link:
            source_links.add(source_for_link)

        # sources.add(metadata.get('source', 'N/A'))

    context = "\n---\n".join(context_parts)

    prompt = f"""You are RAGnarok, an AI assistant for the Lightcast Knowledge Base.
Your goal is to provide concise, accurate, and conversational answers to the user's question based ONLY on the provided context.
If the answer cannot be found in the provided context, state that you don't know or that the information is not available in the knowledge base.
Always include the source URL(s) from the context at the end of your answer, formatted as markdown links. If the source is a local file, state "Source: [Filename]".

Context from Lightcast Knowledge Base:
---
{context}
---

User Question: {query}

Answer:
"""
    st.sidebar.text(f"Prompt length: {len(prompt)} characters")

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False
        },
        "options": {"wait_for_model": True, "use_cache": False}
    }

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and 'generated_text' in result[0]:
            answer = result[0]['generated_text'].strip()
        else:
            answer = f"Received unexpected response from Hugging Face API: {result}"

        if source_links:
            final_sources_display = set()
            for metadata_item in retrieved_metadatas:
                src = metadata_item.get('source', 'N/A')
                if src.startswith('file://'):
                    final_sources_display.add(f"Local File: {os.path.basename(src.replace('file://', ''))}")
                else:
                    final_sources_display.add(f"[{src}]({src})")

            answer += "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in final_sources_display])

        return answer, list(source_links)
    except requests.exceptions.RequestException as e:
        st.error(f"An HTTP error occurred with Hugging Face API: {e}. "
                 f"Response content: {response.text if 'response' in locals() else 'No response received'}")
        return f"An HTTP error occurred with Hugging Face API: {e}", []
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from Hugging Face API response: {e}. "
                 f"Response text: {response.text if 'response' in locals() else 'No response received'}")
        return f"Error decoding JSON from Hugging Face API response: {e}. Response: {response.text}", []
    except Exception as e:
        st.error(f"An unexpected error occurred during generation: {e}")
        return f"An unexpected error occurred during generation: {e}", []
