from __future__ import annotations

from typing import List, Sequence

from langchain_core.documents import Document as LCDocument

from .models import Chunk, Document


def as_langchain_documents(
    items: Sequence[object],
    *,
    chunk_size: int,
    chunk_overlap: int,
    chunk_documents: bool = True,
) -> List[LCDocument]:
    """Normalize canonical documents and chunks into LangChain documents."""

    normalized: List[LCDocument] = []
    for item in items:
        if isinstance(item, LCDocument):
            normalized.append(item)
        elif isinstance(item, Chunk):
            normalized.append(item.to_langchain())
        elif isinstance(item, Document):
            if chunk_documents:
                chunks = item.chunk(chunk_size, chunk_overlap)
                normalized.extend(chunk.to_langchain() for chunk in chunks)
            else:
                normalized.append(item.to_langchain())
        else:
            raise TypeError(f"Unsupported document type for ingestion: {type(item)!r}")
    return normalized
