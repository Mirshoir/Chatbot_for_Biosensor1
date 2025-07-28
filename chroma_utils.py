import os
import uuid
import streamlit as st
from chromadb import PersistentClient

# Define the persistent storage directory for ChromaDB
DATA_STORAGE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(DATA_STORAGE_DIR, "chroma_data")


@st.cache_resource
def get_chroma_client():
    """
    Returns a cached instance of the ChromaDB PersistentClient.
    """
    return PersistentClient(path=CHROMA_PATH)


def get_or_create_collection(collection_name="chat_history"):
    """
    Gets an existing collection by name or creates it if it doesn't exist.
    """
    client = get_chroma_client()
    try:
        return client.get_collection(name=collection_name)
    except Exception:
        return client.create_collection(name=collection_name)


def add_message_to_history(collection, user_message, bot_message, metadata=None):
    """
    Adds a user and bot message pair to the ChromaDB collection.
    """
    user_id = str(uuid.uuid4())
    bot_id = str(uuid.uuid4())

    # Store user message
    collection.add(
        documents=[user_message],
        metadatas=[metadata or {"role": "user"}],
        ids=[user_id]
    )
    # Store bot message
    collection.add(
        documents=[bot_message],
        metadatas=[metadata or {"role": "bot"}],
        ids=[bot_id]
    )


def get_last_n_messages(collection, n=10):
    """
    Retrieves the last n messages from the chat history stored in ChromaDB.
    """
    all_docs = collection.get(include=["documents", "metadatas", "ids"])
    if not all_docs or "documents" not in all_docs:
        return []

    docs = all_docs["documents"]
    metadatas = all_docs["metadatas"]

    # Combine and retrieve the last n messages (no inherent order in UUIDs)
    messages = []
    for doc, meta in zip(docs, metadatas):
        messages.append({"text": doc, "role": meta.get("role", "unknown")})

    return messages[-n:]
