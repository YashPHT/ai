from __future__ import annotations

import sqlite3
from pathlib import Path

from langchain.schema import Document as LCDocument

from ingestion import (
    PDFIngestor,
    StructuredDataIngestor,
    WebsiteIngestor,
    as_langchain_documents,
    Document as CanonicalDocument,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_document_chunking_metadata() -> None:
    document = CanonicalDocument(
        id="doc-1",
        text="Line one for testing.\nLine two continues the sample text.",
        source_type="test",
    )
    chunks = document.chunk(chunk_size=25, chunk_overlap=5)

    assert len(chunks) >= 2
    first = chunks[0]
    assert first.metadata["document_id"] == document.id
    assert first.metadata["chunk_number"] == 1
    assert first.chunk_count == len(chunks)


def test_pdf_ingestion_extracts_text_and_persists(tmp_path: Path) -> None:
    pdf_path = FIXTURES_DIR / "sample.pdf"
    ingestor = PDFIngestor(chunk_size=120, chunk_overlap=10, storage_dir=tmp_path)

    result = ingestor.ingest(pdf_path)

    assert result.documents, "Expected PDF ingestion to yield documents"
    assert result.chunks, "Expected PDF ingestion to produce chunks"
    assert any("Sample PDF" in chunk.text for chunk in result.chunks)

    slug = "sample"
    chunks_path = tmp_path / f"{slug}_chunks.jsonl"
    documents_path = tmp_path / f"{slug}_documents.jsonl"
    assert chunks_path.exists()
    assert documents_path.exists()

    stored_lines = chunks_path.read_text(encoding="utf-8").splitlines()
    assert len(stored_lines) == len(result.chunks)


def test_website_ingestion_cleans_html(tmp_path: Path) -> None:
    html = """
    <html lang="en">
        <head><title>Example Page</title></head>
        <body>
            <h1>Heading</h1>
            <p>Primary paragraph with <strong>important</strong> text.</p>
            <script>console.log('ignored');</script>
        </body>
    </html>
    """
    url = "https://Example.com/some/path?b=2&a=1#section"
    ingestor = WebsiteIngestor(chunk_size=80, chunk_overlap=10, storage_dir=tmp_path)

    result = ingestor.ingest(url, html=html)

    assert result.documents and len(result.documents) == 1
    document = result.documents[0]
    assert document.uri == "https://example.com/some/path?a=1&b=2"
    assert "Heading" in document.text
    assert document.metadata["title"] == "Example Page"
    assert document.metadata["language"] == "en"

    slug = "https-example-com-some-path-a-1-b-2"
    chunks_path = tmp_path / f"{slug}_chunks.jsonl"
    assert chunks_path.exists()


def test_structured_csv_ingestion(tmp_path: Path) -> None:
    csv_path = FIXTURES_DIR / "sample.csv"
    ingestor = StructuredDataIngestor(chunk_size=120, chunk_overlap=20, storage_dir=tmp_path)

    result = ingestor.ingest_csv(csv_path)

    assert result.documents and len(result.documents) == 1
    document = result.documents[0]
    assert document.metadata["format"] == "csv"
    assert document.metadata["row_count"] == 3
    assert document.schema == {"columns": [{"name": "id"}, {"name": "name"}, {"name": "role"}]}

    slug = "sample"
    chunks_path = tmp_path / "csv" / f"{slug}_chunks.jsonl"
    assert chunks_path.exists()


def test_structured_sql_ingestion(tmp_path: Path) -> None:
    db_path = tmp_path / "structured.db"
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT)"
    )
    cursor.executemany(
        "INSERT INTO employees (name, department) VALUES (?, ?)",
        [
            ("Alice", "Engineering"),
            ("Bob", "Finance"),
            ("Charlie", "Sales"),
        ],
    )
    connection.commit()
    connection.close()

    ingestor = StructuredDataIngestor(chunk_size=150, chunk_overlap=30, storage_dir=tmp_path)
    result = ingestor.ingest_sql(
        f"sqlite:///{db_path}",
        table="employees",
        columns=["id", "name", "department"],
        limit=2,
    )

    assert result.documents
    document = result.documents[0]
    assert document.metadata["row_count"] == 2
    assert document.metadata["columns"] == ["id", "name", "department"]
    assert document.schema
    first_column = document.schema["columns"][0]
    assert first_column["name"] == "id"

    chunks_path = tmp_path / "sql" / "employees_chunks.jsonl"
    assert chunks_path.exists()


def test_as_langchain_documents_supports_mixed_inputs() -> None:
    canonical = CanonicalDocument(
        id="doc-2",
        text="Sentence one. Sentence two is here.",
        source_type="test",
    )
    chunk = canonical.chunk(chunk_size=10, chunk_overlap=0)[0]
    lc_document = LCDocument(page_content="Existing chunk", metadata={"chunk_id": "chunk-1"})

    chunked_conversion = as_langchain_documents(
        [canonical],
        chunk_size=10,
        chunk_overlap=0,
    )
    assert len(chunked_conversion) > 1

    raw_conversion = as_langchain_documents(
        [canonical],
        chunk_size=10,
        chunk_overlap=0,
        chunk_documents=False,
    )
    assert len(raw_conversion) == 1

    mixed_conversion = as_langchain_documents(
        [lc_document, chunk],
        chunk_size=10,
        chunk_overlap=0,
    )
    assert len(mixed_conversion) == 2
    assert mixed_conversion[0].metadata["chunk_id"] == "chunk-1"
