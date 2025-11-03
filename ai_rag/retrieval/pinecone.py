from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Sequence

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .types import RetrievalResult


class PineconeIndexManager:
    """Utility wrapper around the Pinecone client for index management."""

    def __init__(
        self,
        *,
        index_name: str,
        dimension: int,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        namespace: Optional[str] = "default",
        client: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.index_name = index_name
        self.dimension = dimension
        self.api_key = api_key
        self.environment = environment
        self.namespace = namespace or "default"
        self._client = client
        self._index = None
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Client helpers
    # ------------------------------------------------------------------
    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:  # pragma: no cover - exercised in integration only
            import pinecone  # type: ignore
        except ImportError as exc:  # pragma: no cover - defensive branch
            raise RuntimeError(
                "The 'pinecone' package is required to use the Pinecone retriever"
            ) from exc

        if hasattr(pinecone, "Pinecone"):
            if not self.api_key:
                raise RuntimeError("Pinecone API key is required to initialize client")
            self._client = pinecone.Pinecone(api_key=self.api_key)
        else:  # legacy client
            if not (self.api_key and self.environment):
                raise RuntimeError("Pinecone API key and environment are required")
            pinecone.init(api_key=self.api_key, environment=self.environment)
            self._client = pinecone
        return self._client

    def _resolve_index(self) -> Any:
        if self._index is not None:
            return self._index

        client = self._ensure_client()
        if hasattr(client, "Index"):
            self._index = client.Index(self.index_name)
        elif hasattr(client, "index"):
            self._index = client.index(self.index_name)
        else:  # pragma: no cover - defensive branch
            raise RuntimeError("Pinecone client does not expose an Index accessor")
        return self._index

    # ------------------------------------------------------------------
    # Index management operations
    # ------------------------------------------------------------------
    def list_indexes(self) -> List[str]:
        client = self._ensure_client()
        if not hasattr(client, "list_indexes"):
            return []

        response = client.list_indexes()
        if hasattr(response, "names"):
            response = response.names()
        if isinstance(response, dict):
            if "indexes" in response:
                return [entry.get("name") for entry in response["indexes"] if entry.get("name")]
            if "names" in response:
                return list(response["names"])
        if isinstance(response, Sequence):
            return list(response)
        return []

    def create_index(self, metric: str = "cosine", **kwargs: Any) -> bool:
        client = self._ensure_client()
        if self.index_name in self.list_indexes():
            self.logger.debug("Pinecone index '%s' already exists", self.index_name)
            return False

        if not hasattr(client, "create_index"):
            raise RuntimeError("Pinecone client does not support create_index")

        client.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=metric,
            **kwargs,
        )
        self.logger.info("Created Pinecone index '%s'", self.index_name)
        return True

    def delete_index(self) -> bool:
        client = self._ensure_client()
        if not hasattr(client, "delete_index"):
            raise RuntimeError("Pinecone client does not support delete_index")

        if self.index_name not in self.list_indexes():
            self.logger.debug("Pinecone index '%s' does not exist", self.index_name)
            return False

        client.delete_index(self.index_name)
        self._index = None
        self.logger.info("Deleted Pinecone index '%s'", self.index_name)
        return True

    def ensure_index(self, metric: str = "cosine", **kwargs: Any) -> Any:
        if self.index_name not in self.list_indexes():
            self.create_index(metric=metric, **kwargs)
        return self._resolve_index()

    def get_index_stats(self) -> Dict[str, Any]:
        """Retrieve index statistics including vector count."""
        index = self._resolve_index()
        if not hasattr(index, "describe_index_stats"):
            return {"total_vector_count": 0, "namespaces": {}}
        
        try:
            stats = index.describe_index_stats()
            if isinstance(stats, dict):
                return stats
            return {
                "total_vector_count": getattr(stats, "total_vector_count", 0),
                "namespaces": getattr(stats, "namespaces", {}),
            }
        except Exception as error:
            self.logger.warning("Failed to get index stats: %s", error)
            return {"total_vector_count": 0, "namespaces": {}}

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------
    def upsert(self, vectors: Sequence[Dict[str, Any]]) -> Any:
        if not vectors:
            return None

        index = self._resolve_index()
        if not hasattr(index, "upsert"):
            raise RuntimeError("Pinecone index handle does not expose upsert")

        return index.upsert(vectors=vectors, namespace=self.namespace)

    def query(
        self,
        *,
        vector: Sequence[float],
        top_k: int,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> Any:
        index = self._resolve_index()
        if not hasattr(index, "query"):
            raise RuntimeError("Pinecone index handle does not expose query")

        params = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": include_metadata,
            "namespace": kwargs.pop("namespace", self.namespace),
        }
        params.update(kwargs)
        return index.query(**params)


class PineconeEmbeddingPipeline:
    """Embeds and upserts documents into Pinecone in batches."""

    def __init__(
        self,
        *,
        index_manager: PineconeIndexManager,
        embeddings: Any,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        batch_size: int = 32,
        namespace: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.index_manager = index_manager
        self.embeddings = embeddings
        self.batch_size = max(1, batch_size)
        self.namespace = namespace or index_manager.namespace
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def run(self, documents: Sequence[Document]) -> int:
        """Process and upsert documents into Pinecone (alias for upsert_documents)."""
        return self.upsert_documents(documents)

    def upsert_documents(self, documents: Sequence[Document]) -> int:
        if not documents:
            return 0

        chunks = self._splitter.split_documents(list(documents))
        if not chunks:
            return 0

        total_upserted = 0
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            texts = [doc.page_content for doc in batch]

            if not hasattr(self.embeddings, "embed_documents"):
                raise RuntimeError("Embeddings implementation lacks 'embed_documents'")

            vectors = self.embeddings.embed_documents(texts)
            if len(vectors) != len(batch):
                raise RuntimeError("Embeddings output size does not match input batch")

            payload = []
            for offset, (doc, vector) in enumerate(zip(batch, vectors)):
                metadata = dict(doc.metadata or {})
                metadata["text"] = doc.page_content
                metadata.setdefault("source", metadata.get("source_id", "unknown"))
                metadata.setdefault("chunk_index", start + offset)

                chunk_id = metadata.get("id") or metadata.get("chunk_id")
                if not chunk_id:
                    source = metadata.get("source", "doc")
                    chunk_id = f"{source}-{start + offset}-{uuid.uuid4().hex[:8]}"
                metadata["chunk_id"] = chunk_id

                payload.append(
                    {
                        "id": str(chunk_id),
                        "values": vector,
                        "metadata": metadata,
                    }
                )

            self.index_manager.upsert(payload)
            total_upserted += len(payload)
            self.logger.debug(
                "Upserted %s chunks into Pinecone (total=%s)", len(payload), total_upserted
            )

        return total_upserted


class PineconeRetriever:
    """Retrieves relevant content from Pinecone using embeddings."""

    retriever_name = "pinecone"

    def __init__(
        self,
        *,
        index_manager: PineconeIndexManager,
        embeddings: Any,
        top_k: int = 5,
        namespace: Optional[str] = None,
    ) -> None:
        self.index_manager = index_manager
        self.embeddings = embeddings
        self.top_k = max(1, top_k)
        self.namespace = namespace or index_manager.namespace

    def retrieve(self, query: str, k: Optional[int] = None) -> List[RetrievalResult]:
        if not query.strip():
            return []

        if not hasattr(self.embeddings, "embed_query"):
            raise RuntimeError("Embeddings implementation lacks 'embed_query'")

        vector = self.embeddings.embed_query(query)
        limit = max(1, k or self.top_k)
        response = self.index_manager.query(
            vector=vector,
            top_k=limit,
            include_metadata=True,
            namespace=self.namespace,
        )

        matches = self._extract_matches(response)

        results: List[RetrievalResult] = []
        for match in matches:
            metadata = self._extract_metadata(match)
            score = self._extract_score(match)
            text = metadata.pop("text", metadata.pop("page_content", ""))
            if not text:
                continue
            results.append(
                RetrievalResult(
                    content=text,
                    score=score,
                    metadata=metadata,
                    retriever=self.retriever_name,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------
    def _extract_matches(self, response: Any) -> List[Any]:
        if response is None:
            return []
        if isinstance(response, dict):
            matches = response.get("matches", [])
            if isinstance(matches, list):
                return matches
            return []
        matches = getattr(response, "matches", None)
        if isinstance(matches, list):
            return matches
        return []

    def _extract_metadata(self, match: Any) -> Dict[str, Any]:
        if isinstance(match, dict):
            metadata = match.get("metadata", {})
            meta = dict(metadata) if isinstance(metadata, dict) else {}
            match_id = match.get("id")
        else:
            metadata = getattr(match, "metadata", {})
            meta = dict(metadata) if isinstance(metadata, dict) else {}
            match_id = getattr(match, "id", None)
        if match_id is not None:
            meta.setdefault("id", match_id)
        similarity = self._extract_score(match)
        meta.setdefault("similarity", similarity)
        meta.setdefault("retriever", self.retriever_name)
        return meta

    def _extract_score(self, match: Any) -> float:
        if isinstance(match, dict):
            value = match.get("score") or match.get("values")
        else:
            value = getattr(match, "score", None) or getattr(match, "values", None)
        try:
            if isinstance(value, (list, tuple)) and value:
                return float(value[0])
            if value is None:
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0


__all__ = [
    "PineconeIndexManager",
    "PineconeEmbeddingPipeline",
    "PineconeRetriever",
]
