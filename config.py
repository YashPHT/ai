import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RAGConfig:
    google_api_key: str
    environment: str = "local"
    enable_auth: bool = False
    retriever_top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    vector_store_path: str = "./chroma_db"
    enable_graph_retriever: bool = False
    graph_retriever_top_k: int = 3
    enable_telemetry: bool = False
    
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
            enable_graph_retriever=os.getenv("ENABLE_GRAPH_RETRIEVER", "false").lower() == "true",
            graph_retriever_top_k=int(os.getenv("GRAPH_RETRIEVER_TOP_K", "3")),
            enable_telemetry=os.getenv("ENABLE_TELEMETRY", "false").lower() == "true",
        )
    
    def validate(self) -> tuple[bool, Optional[str]]:
        if not self.google_api_key:
            return False, "GOOGLE_API_KEY is required"
        if self.retriever_top_k <= 0:
            return False, "RETRIEVER_TOP_K must be positive"
        if self.chunk_size <= 0:
            return False, "CHUNK_SIZE must be positive"
        if self.graph_retriever_top_k <= 0:
            return False, "GRAPH_RETRIEVER_TOP_K must be positive"
        return True, None
