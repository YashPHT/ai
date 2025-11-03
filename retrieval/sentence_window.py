from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from langchain.schema import Document

from .types import RetrievalResult



def _split_sentences(text: str) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    # Split on sentence boundaries while preserving abbreviations reasonably.
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def _tokenize(text: str) -> Set[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {token for token in tokens if len(token) > 1}


@dataclass(slots=True)
class SentenceWindowRecord:
    key: str
    sentence: str
    context_before: List[str]
    context_after: List[str]
    metadata: Dict[str, Any]
    tokens: Set[str] = field(default_factory=set)

    def render_context(self) -> str:
        segments = [*self.context_before, self.sentence, *self.context_after]
        return " ".join(segment.strip() for segment in segments if segment)


class SentenceWindowRetriever:
    """Retrieves sentences with configurable context windows around matches."""

    retriever_name = "sentence_window"

    def __init__(self, *, window_size: int = 1, top_k: int = 5) -> None:
        self.window_size = max(0, window_size)
        self.top_k = max(1, top_k)
        self._records: List[SentenceWindowRecord] = []
        self._indexed_documents: Set[str] = set()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._records.clear()
        self._indexed_documents.clear()

    def index_documents(self, documents: Sequence[Document]) -> None:
        for document in documents:
            key = self._document_key(document)
            if key in self._indexed_documents:
                continue

            sentences = _split_sentences(document.page_content)
            if not sentences:
                continue

            metadata = dict(document.metadata or {})
            metadata.setdefault("source", metadata.get("source_id", "unknown"))

            for idx, sentence in enumerate(sentences):
                before_start = max(0, idx - self.window_size)
                before = sentences[before_start:idx]
                after_end = idx + self.window_size + 1
                after = sentences[idx + 1 : after_end]

                record = SentenceWindowRecord(
                    key=f"{key}-{idx}",
                    sentence=sentence,
                    context_before=before,
                    context_after=after,
                    metadata=metadata,
                    tokens=_tokenize(sentence),
                )
                self._records.append(record)

            self._indexed_documents.add(key)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: Optional[int] = None) -> List[RetrievalResult]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        limit = max(1, k or self.top_k)
        scored: List[tuple[float, SentenceWindowRecord]] = []

        for record in self._records:
            overlap = len(query_tokens & record.tokens)
            if overlap == 0:
                continue
            union = len(query_tokens | record.tokens)
            score = overlap / union if union else 0.0
            if score <= 0:
                continue
            scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_records = scored[:limit]

        results: List[RetrievalResult] = []
        for score, record in top_records:
            metadata = dict(record.metadata)
            metadata.update(
                {
                    "sentence": record.sentence,
                    "context_before": record.context_before,
                    "context_after": record.context_after,
                }
            )
            results.append(
                RetrievalResult(
                    content=record.render_context(),
                    score=score,
                    metadata=metadata,
                    retriever=self.retriever_name,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _document_key(self, document: Document) -> str:
        metadata = document.metadata or {}
        source = metadata.get("source") or metadata.get("source_id") or "doc"
        page = metadata.get("page") or metadata.get("id") or 0
        return f"{source}:{page}"


__all__ = ["SentenceWindowRetriever"]
