from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )

    google_api_key: str = Field("", description="Google API key for Gemini")
    gemini_api_key: Optional[str] = Field(None, description="Alias for Google API key")
    pinecone_api_key: Optional[str] = Field(None, description="Pinecone API key")
    pinecone_environment: Optional[str] = Field(None, description="Pinecone environment")
    pinecone_index_name: str = Field(
        "retrieval-suite", description="Pinecone index name"
    )

    environment: str = Field("local", description="Deployment environment")
    enable_auth: bool = Field(False, description="Enable authentication")

    retriever_top_k: int = Field(5, description="Number of documents to retrieve")
    chunk_size: int = Field(1000, description="Document chunk size")
    chunk_overlap: int = Field(200, description="Overlap between chunks")

    enable_pinecone_retriever: bool = Field(False, description="Enable Pinecone retriever")
    pinecone_namespace: str = Field("default", description="Pinecone namespace")
    pinecone_dimension: int = Field(768, description="Embedding dimension")
    pinecone_metric: str = Field("cosine", description="Similarity metric")
    pinecone_top_k: int = Field(5, description="Top K for Pinecone")
    pinecone_batch_size: int = Field(32, description="Batch size for Pinecone")

    enable_sentence_window_retriever: bool = Field(
        False, description="Enable sentence window retriever"
    )
    sentence_window_size: int = Field(1, description="Sentence window size")
    sentence_retriever_top_k: int = Field(5, description="Top K for sentence retriever")

    enable_graph_retriever: bool = Field(False, description="Enable graph retriever")
    graph_retriever_top_k: int = Field(3, description="Top K for graph retriever")
    graph_max_depth: int = Field(2, description="Maximum graph depth")

    enable_telemetry: bool = Field(False, description="Enable telemetry")
    fusion_token_budget: int = Field(1200, description="Token budget for fusion")
    fusion_rrf_k: int = Field(60, description="RRF K parameter")
    enable_fusion_reranker: bool = Field(True, description="Enable fusion reranker")
    fusion_reranker_weight: float = Field(0.35, description="Reranker weight")

    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", description="Embedding model"
    )
    gemini_model: str = Field("gemini-pro", description="Gemini model name")

    generation_prompt: str = Field(
        """You are a helpful enterprise Q&A assistant. Answer the user's question based on the context provided.
If the context does not contain the answer, state that you cannot answer the question.
\n\n
Context:
{context}
\n\n
Question:
{question}
\n\n
Answer:
""",
        description="Generation prompt template",
    )

    @field_validator("google_api_key", mode="before")
    @classmethod
    def validate_api_key(cls, v: Optional[str], info) -> str:
        if not v:
            gemini_key = info.data.get("gemini_api_key")
            if gemini_key:
                return gemini_key
            raise ValueError("Either GOOGLE_API_KEY or GEMINI_API_KEY must be provided")
        return v

    @field_validator("fusion_reranker_weight")
    @classmethod
    def validate_reranker_weight(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("fusion_reranker_weight must be between 0 and 1")
        return v

    def validate_pinecone_config(self) -> tuple[bool, Optional[str]]:
        if self.enable_pinecone_retriever and not self.pinecone_api_key:
            return False, "PINECONE_API_KEY is required when Pinecone retriever is enabled"
        return True, None

    def validate(self) -> tuple[bool, Optional[str]]:
        """Compatibility helper returning (is_valid, error_message)."""

        checks = [
            (bool(self.google_api_key), "GOOGLE_API_KEY is required"),
            (self.retriever_top_k > 0, "RETRIEVER_TOP_K must be positive"),
            (self.chunk_size > 0, "CHUNK_SIZE must be positive"),
            (self.pinecone_top_k > 0, "PINECONE_TOP_K must be positive"),
            (self.pinecone_dimension > 0, "PINECONE_DIMENSION must be positive"),
            (self.pinecone_batch_size > 0, "PINECONE_BATCH_SIZE must be positive"),
            (self.sentence_window_size >= 0, "SENTENCE_WINDOW_SIZE must be non-negative"),
            (
                self.sentence_retriever_top_k > 0,
                "SENTENCE_RETRIEVER_TOP_K must be positive",
            ),
            (self.graph_retriever_top_k > 0, "GRAPH_RETRIEVER_TOP_K must be positive"),
            (self.graph_max_depth > 0, "GRAPH_MAX_DEPTH must be positive"),
            (self.fusion_token_budget >= 0, "FUSION_TOKEN_BUDGET must be non-negative"),
            (self.fusion_rrf_k > 0, "FUSION_RRF_K must be positive"),
            (
                0 <= self.fusion_reranker_weight <= 1,
                "FUSION_RERANKER_WEIGHT must be between 0 and 1",
            ),
            (
                not self.enable_pinecone_retriever or bool(self.pinecone_api_key),
                "PINECONE_API_KEY is required when Pinecone retriever is enabled",
            ),
            (bool(self.generation_prompt), "GENERATION_PROMPT is required"),
        ]

        for condition, error_message in checks:
            if not condition:
                return False, error_message
        return True, None

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment (for backwards compatibility)."""
        return cls()


settings = Settings()
