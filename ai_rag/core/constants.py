"""Centralized constants for the Enterprise RAG system."""

from typing import Final

DEFAULT_PINECONE_INDEX: Final[str] = "retrieval-suite"
DEFAULT_PINECONE_NAMESPACE: Final[str] = "default"
DEFAULT_EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GEMINI_MODEL: Final[str] = "gemini-pro"
DEFAULT_CHUNK_SIZE: Final[int] = 1000
DEFAULT_CHUNK_OVERLAP: Final[int] = 200
DEFAULT_RETRIEVER_TOP_K: Final[int] = 5
