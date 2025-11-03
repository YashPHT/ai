from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import Chunk, Document, IngestionResult
from .utils import ensure_directory


def _to_serializable(record: dict) -> dict:
    serialized = {}
    for key, value in record.items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


def write_jsonl(path: Path, records: Iterable[dict]) -> Path:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(_to_serializable(record), handle, ensure_ascii=False)
            handle.write("\n")
    return path


def persist_result(storage_dir: Path, stem: str, result: IngestionResult) -> None:
    ensure_directory(storage_dir)
    documents_path = storage_dir / f"{stem}_documents.jsonl"
    chunks_path = storage_dir / f"{stem}_chunks.jsonl"
    write_jsonl(documents_path, (document.to_dict() for document in result.documents))
    write_jsonl(chunks_path, (chunk.to_dict() for chunk in result.chunks))
