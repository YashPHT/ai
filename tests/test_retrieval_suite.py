from __future__ import annotations

from unittest.mock import MagicMock

from langchain.schema import Document

from ai_rag.retrieval import (
    GraphRetriever,
    PineconeEmbeddingPipeline,
    PineconeIndexManager,
    PineconeRetriever,
    SentenceWindowRetriever,
)


def test_pinecone_embedding_pipeline_batches_and_upserts() -> None:
    client = MagicMock()
    index = MagicMock()
    client.Index.return_value = index
    client.list_indexes.return_value = ["retrieval-suite"]

    manager = PineconeIndexManager(
        index_name="retrieval-suite",
        dimension=3,
        client=client,
    )

    embeddings = MagicMock()
    embeddings.embed_documents.side_effect = [
        [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]],
        [[0.3, 0.3, 0.3]],
    ]

    pipeline = PineconeEmbeddingPipeline(
        index_manager=manager,
        embeddings=embeddings,
        batch_size=2,
    )

    documents = [
        Document(page_content="First chunk of text", metadata={"source": "doc1", "page": 1}),
        Document(page_content="Second chunk of text", metadata={"source": "doc2", "page": 1}),
        Document(page_content="Third chunk of text", metadata={"source": "doc3", "page": 1}),
    ]

    upserted = pipeline.upsert_documents(documents)

    assert upserted == len(documents)
    assert index.upsert.call_count == 2

    first_call = index.upsert.call_args_list[0].kwargs["vectors"]
    second_call = index.upsert.call_args_list[1].kwargs["vectors"]

    assert len(first_call) == 2
    assert len(second_call) == 1
    assert all("text" in entry["metadata"] for entry in first_call + second_call)


def test_pinecone_retriever_returns_standardized_results() -> None:
    client = MagicMock()
    index = MagicMock()
    client.Index.return_value = index
    client.list_indexes.return_value = ["retrieval-suite"]

    manager = PineconeIndexManager(
        index_name="retrieval-suite",
        dimension=3,
        client=client,
    )

    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

    index.query.return_value = {
        "matches": [
            {
                "id": "chunk-1",
                "score": 0.92,
                "metadata": {"text": "Doc 1", "source": "source1"},
            },
            {
                "id": "chunk-2",
                "score": 0.81,
                "metadata": {"text": "Doc 2", "source": "source2"},
            },
        ]
    }

    retriever = PineconeRetriever(
        index_manager=manager,
        embeddings=embeddings,
        top_k=2,
    )

    results = retriever.retrieve("cloud architecture", k=2)

    assert len(results) == 2
    assert results[0].retriever == "pinecone"

    doc = results[0].to_document()
    assert doc.metadata["retriever"] == "pinecone"
    assert "score" in doc.metadata
    index.query.assert_called_once()


def test_sentence_window_retriever_returns_context_window() -> None:
    retriever = SentenceWindowRetriever(window_size=1, top_k=2)
    document = Document(
        page_content=(
            "Alpha builds platforms. Beta integrates services. Gamma secures them."
        ),
        metadata={"source": "tech_paper", "page": 4},
    )

    retriever.index_documents([document])
    results = retriever.retrieve("Beta services", k=1)

    assert len(results) == 1
    result = results[0]
    assert "Alpha builds platforms." in result.content
    assert "Gamma secures them." in result.content

    doc = result.to_document()
    assert doc.metadata["retriever"] == "sentence_window"
    assert doc.metadata["sentence"].startswith("Beta integrates services")


def test_graph_retriever_returns_entity_context() -> None:
    retriever = GraphRetriever(max_depth=2, top_k=2)
    document = Document(
        page_content=(
            "Azure integrates with GitHub Actions for CI/CD automation. "
            "GitHub Actions coordinates deployments across Azure regions. "
            "Azure provides enterprise-grade security."
        ),
        metadata={"source": "integration_guide", "page": 7},
    )

    retriever.index_documents([document])
    results = retriever.retrieve("How does Azure work with GitHub?", k=1)

    assert results
    result = results[0]
    assert "Azure" in result.content

    doc = result.to_document()
    assert doc.metadata["retriever"] == "graph"
    assert doc.metadata["entities"]
    assert doc.metadata["graph_path"]
