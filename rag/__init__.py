"""RAG: ChromaDB vectorstore para busca semantica."""
from .vectorstore import get_client, get_or_create_collection
from .indexer import index_all
from .retriever import Retriever

__all__ = ["get_client", "get_or_create_collection", "index_all", "Retriever"]
