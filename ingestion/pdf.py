from __future__ import annotations

from pathlib import Path
from typing import List

from pypdf import PdfReader

from .models import Document, IngestionResult
from .storage import persist_result
from .utils import clean_text, slugify


class PDFIngestor:
    """Ingest PDF documents into the canonical document format."""

    def __init__(
        self,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        storage_dir: Path | str = Path("data/ingestion/pdf"),
        persist: bool = True,
    ) -> None:
        self.chunk_size = max(1, chunk_size)
        self.chunk_overlap = max(0, chunk_overlap)
        self.storage_dir = Path(storage_dir)
        self.persist = persist

    def ingest(self, path: Path | str) -> IngestionResult:
        pdf_path = Path(path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        metadata = reader.metadata or {}
        documents: List[Document] = []

        for index, page in enumerate(reader.pages, start=1):
            extracted_text = page.extract_text() or ""
            cleaned_text = clean_text(extracted_text)
            if not cleaned_text:
                continue

            page_metadata = {
                "source": pdf_path.name,
                "page_number": index,
                "total_pages": total_pages,
                "source_type": "pdf",
            }

            title = getattr(metadata, "title", None)
            author = getattr(metadata, "author", None)
            if title:
                page_metadata["title"] = str(title)
            if author:
                page_metadata["author"] = str(author)

            document = Document(
                id=f"{slugify(pdf_path.stem)}-page-{index}",
                text=cleaned_text,
                source_type="pdf",
                path=str(pdf_path.resolve()),
                metadata=page_metadata,
            )
            documents.append(document)

        result = IngestionResult(
            documents=documents,
            chunks=[
                chunk
                for document in documents
                for chunk in document.chunk(self.chunk_size, self.chunk_overlap)
            ],
            source=str(pdf_path.resolve()),
        )

        if self.persist and documents:
            persist_result(self.storage_dir, slugify(pdf_path.stem), result)

        return result
