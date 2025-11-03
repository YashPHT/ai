from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List
from urllib.parse import urlencode, urlsplit, urlunsplit, parse_qsl


@dataclass(slots=True)
class TextChunk:
    """Chunk of text with start/end positions for provenance."""

    text: str
    start: int
    end: int


def ensure_directory(path: Path) -> None:
    """Ensure a directory exists, creating parents as needed."""

    path.mkdir(parents=True, exist_ok=True)


def clean_text(value: str | None) -> str:
    """Normalize whitespace and unicode artifacts from extracted text."""

    if not value:
        return ""

    text = unicodedata.normalize("NFKC", value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\v\f]+", " ", text)
    lines = []
    for raw_line in text.split("\n"):
        normalized_line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if normalized_line:
            lines.append(normalized_line)
    return "\n".join(lines)


def embedding_ready_text(value: str | None) -> str:
    """Prepare text for embedding generation by lowercasing and cleaning."""

    cleaned = clean_text(value)
    return cleaned.lower()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int = 0) -> List[TextChunk]:
    """Split text into overlapping chunks for downstream indexing."""

    normalized = clean_text(text)
    if not normalized:
        return []

    chunk_size = max(1, chunk_size)
    chunk_overlap = max(0, chunk_overlap)
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size - 1)

    stride = chunk_size - chunk_overlap or chunk_size
    chunks: List[TextChunk] = []
    start = 0
    length = len(normalized)

    while start < length:
        end = min(start + chunk_size, length)
        chunk_text_value = normalized[start:end]
        chunks.append(TextChunk(text=chunk_text_value, start=start, end=end))
        if end >= length:
            break
        start += stride

    return chunks


def slugify(value: str, *, max_length: int = 64) -> str:
    """Create a filesystem- and identifier-safe slug."""

    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-") or "document"
    if len(value) > max_length:
        value = value[:max_length]
    return value


def canonicalize_url(url: str) -> str:
    """Normalize URLs to a canonical form for deduplication."""

    parts = urlsplit(url)
    scheme = parts.scheme.lower() or "http"
    netloc = parts.netloc.lower()
    path = parts.path or "/"
    query_pairs = sorted(parse_qsl(parts.query, keep_blank_values=True))
    query = urlencode(query_pairs)
    return urlunsplit((scheme, netloc, path, query, ""))
