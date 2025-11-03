from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain.schema import Document as LCDocument

from .utils import TextChunk, chunk_text, clean_text


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class Document:
    """Canonical representation of an ingested source document."""

    id: str
    text: str
    source_type: str
    uri: Optional[str] = None
    path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    def __post_init__(self) -> None:
        self.text = clean_text(self.text)

    def chunk(self, chunk_size: int, chunk_overlap: int = 0) -> List["Chunk"]:
        segments = chunk_text(self.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        total = len(segments)
        chunks: List[Chunk] = []
        for index, segment in enumerate(segments):
            chunk_metadata = {**self.metadata}
            chunk_metadata.update(
                {
                    "document_id": self.id,
                    "chunk_index": index,
                    "chunk_number": index + 1,
                    "chunk_count": total,
                    "char_start": segment.start,
                    "char_end": segment.end,
                }
            )
            chunk = Chunk(
                id=f"{self.id}-chunk-{index + 1}",
                document_id=self.id,
                text=segment.text,
                chunk_index=index,
                chunk_count=total,
                metadata=chunk_metadata,
            )
            chunks.append(chunk)
        return chunks

    def to_langchain(self) -> LCDocument:
        metadata = {**self.metadata}
        metadata.setdefault("document_id", self.id)
        metadata.setdefault("source_type", self.source_type)
        if self.uri:
            metadata.setdefault("uri", self.uri)
        if self.path:
            metadata.setdefault("path", self.path)
        if self.schema is not None:
            metadata.setdefault("schema", self.schema)
        metadata.setdefault("created_at", self.created_at.isoformat())
        metadata.setdefault("updated_at", self.updated_at.isoformat())
        return LCDocument(page_content=self.text, metadata=metadata)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "id": self.id,
            "text": self.text,
            "source_type": self.source_type,
            "uri": self.uri,
            "path": self.path,
            "metadata": self.metadata,
            "schema": self.schema,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        return {key: value for key, value in payload.items() if value is not None}


@dataclass(slots=True)
class Chunk:
    """Chunk of text produced from a canonical document."""

    id: str
    document_id: str
    text: str
    chunk_index: int
    chunk_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_now)

    def __post_init__(self) -> None:
        self.text = clean_text(self.text)

    def to_langchain(self) -> LCDocument:
        metadata = {**self.metadata}
        metadata.setdefault("chunk_id", self.id)
        metadata.setdefault("document_id", self.document_id)
        metadata.setdefault("chunk_index", self.chunk_index)
        metadata.setdefault("chunk_count", self.chunk_count)
        metadata.setdefault("created_at", self.created_at.isoformat())
        return LCDocument(page_content=self.text, metadata=metadata)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }
        return payload


@dataclass(slots=True)
class IngestionResult:
    """Container for ingestion outputs."""

    documents: List[Document]
    chunks: List[Chunk]
    source: str

    def to_langchain(self) -> List[LCDocument]:
        return [chunk.to_langchain() for chunk in self.chunks]

    @property
    def is_empty(self) -> bool:
        return not self.documents
