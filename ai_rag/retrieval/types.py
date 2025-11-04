from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from langchain_core.documents import Document


@dataclass(slots=True)
class RetrievalResult:
    """Standard representation of a retriever response.

    All retrievers return their results using this schema so downstream
    consumers such as the fusion pipeline can treat the outputs
    uniformly by converting them into LangChain ``Document`` objects.
    """

    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    retriever: str = "unknown"

    def to_document(self) -> Document:
        """Convert the retrieval result into a LangChain ``Document``."""

        doc_metadata: Dict[str, Any] = dict(self.metadata) if self.metadata else {}
        doc_metadata.setdefault("retriever", self.retriever)
        doc_metadata["score"] = self.score
        return Document(page_content=self.content, metadata=doc_metadata)


__all__ = ["RetrievalResult"]
