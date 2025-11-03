import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

DEFAULT_GENERATION_PROMPT = """
You are a helpful enterprise Q&A assistant. Answer the user's question based on the context provided.
If the context does not contain the answer, state that you cannot answer the question.
\n\n
Context:
{context}
\n\n
Question:
{question}
\n\n
Answer:
"""


@dataclass
class RAGConfig:
    google_api_key: str
    environment: str = "local"
    enable_auth: bool = False
    retriever_top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    vector_store_path: str = "./chroma_db"
    enable_pinecone_retriever: bool = False
    pinecone_api_key: str = ""
    pinecone_environment: str = ""
    pinecone_index_name: str = "retrieval-suite"
    pinecone_namespace: str = "default"
    pinecone_dimension: int = 768
    pinecone_metric: str = "cosine"
    pinecone_top_k: int = 5
    pinecone_batch_size: int = 32
    enable_sentence_window_retriever: bool = False
    sentence_window_size: int = 1
    sentence_retriever_top_k: int = 5
    enable_graph_retriever: bool = False
    graph_retriever_top_k: int = 3
    graph_max_depth: int = 2
    enable_telemetry: bool = False
    fusion_token_budget: int = 1200
    fusion_rrf_k: int = 60
    enable_fusion_reranker: bool = True
    fusion_reranker_weight: float = 0.35
    generation_prompt: str = field(default_factory=lambda: DEFAULT_GENERATION_PROMPT)

    @classmethod
    def from_env(cls) -> "RAGConfig":
        return cls(
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            environment=os.getenv("ENVIRONMENT", "local"),
            enable_auth=os.getenv("ENABLE_AUTH", "false").lower() == "true",
            retriever_top_k=int(os.getenv("RETRIEVER_TOP_K", "5")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            vector_store_path=os.getenv("VECTOR_STORE_PATH", "./chroma_db"),
            enable_pinecone_retriever=os.getenv("ENABLE_PINECONE_RETRIEVER", "false").lower() == "true",
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            pinecone_environment=os.getenv("PINECONE_ENVIRONMENT", ""),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "retrieval-suite"),
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "default"),
            pinecone_dimension=int(os.getenv("PINECONE_DIMENSION", "768")),
            pinecone_metric=os.getenv("PINECONE_METRIC", "cosine"),
            pinecone_top_k=int(os.getenv("PINECONE_TOP_K", "5")),
            pinecone_batch_size=int(os.getenv("PINECONE_BATCH_SIZE", "32")),
            enable_sentence_window_retriever=os.getenv("ENABLE_SENTENCE_WINDOW_RETRIEVER", "false").lower() == "true",
            sentence_window_size=int(os.getenv("SENTENCE_WINDOW_SIZE", "1")),
            sentence_retriever_top_k=int(os.getenv("SENTENCE_RETRIEVER_TOP_K", "5")),
            enable_graph_retriever=os.getenv("ENABLE_GRAPH_RETRIEVER", "false").lower() == "true",
            graph_retriever_top_k=int(os.getenv("GRAPH_RETRIEVER_TOP_K", "3")),
            graph_max_depth=int(os.getenv("GRAPH_MAX_DEPTH", "2")),
            enable_telemetry=os.getenv("ENABLE_TELEMETRY", "false").lower() == "true",
            fusion_token_budget=int(os.getenv("FUSION_TOKEN_BUDGET", "1200")),
            fusion_rrf_k=int(os.getenv("FUSION_RRF_K", "60")),
            enable_fusion_reranker=os.getenv("ENABLE_FUSION_RERANKER", "true").lower() == "true",
            fusion_reranker_weight=float(os.getenv("FUSION_RERANKER_WEIGHT", "0.35")),
            generation_prompt=os.getenv("GENERATION_PROMPT", DEFAULT_GENERATION_PROMPT),
        )

    def validate(self) -> tuple[bool, Optional[str]]:
        if not self.google_api_key:
            return False, "GOOGLE_API_KEY is required"
        if self.retriever_top_k <= 0:
            return False, "RETRIEVER_TOP_K must be positive"
        if self.chunk_size <= 0:
            return False, "CHUNK_SIZE must be positive"
        if self.pinecone_top_k <= 0:
            return False, "PINECONE_TOP_K must be positive"
        if self.pinecone_dimension <= 0:
            return False, "PINECONE_DIMENSION must be positive"
        if self.pinecone_batch_size <= 0:
            return False, "PINECONE_BATCH_SIZE must be positive"
        if self.sentence_window_size < 0:
            return False, "SENTENCE_WINDOW_SIZE must be non-negative"
        if self.sentence_retriever_top_k <= 0:
            return False, "SENTENCE_RETRIEVER_TOP_K must be positive"
        if self.graph_retriever_top_k <= 0:
            return False, "GRAPH_RETRIEVER_TOP_K must be positive"
        if self.graph_max_depth <= 0:
            return False, "GRAPH_MAX_DEPTH must be positive"
        if self.fusion_token_budget < 0:
            return False, "FUSION_TOKEN_BUDGET must be non-negative"
        if self.fusion_rrf_k <= 0:
            return False, "FUSION_RRF_K must be positive"
        if not 0 <= self.fusion_reranker_weight <= 1:
            return False, "FUSION_RERANKER_WEIGHT must be between 0 and 1"
        if self.enable_pinecone_retriever and not self.pinecone_api_key:
            return False, "PINECONE_API_KEY is required when Pinecone retriever is enabled"
        if not self.generation_prompt:
            return False, "GENERATION_PROMPT is required"
        return True, None
