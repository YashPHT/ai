import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import Document

from config import RAGConfig
from rag_workflow import RAGWorkflow, RAGState


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(page_content="Test content 1", metadata={"source": "test1.pdf", "page": 1}),
        Document(page_content="Test content 2", metadata={"source": "test2.pdf", "page": 2}),
    ]


@pytest.fixture
def mock_config() -> RAGConfig:
    return RAGConfig(
        google_api_key="test_key",
        environment="test",
        retriever_top_k=2,
        graph_retriever_top_k=2,
    )


@pytest.fixture
def mock_workflow(mock_config: RAGConfig, sample_documents: list[Document]) -> RAGWorkflow:
    llm = MagicMock()
    embeddings = MagicMock()
    vector_store = MagicMock()

    vector_store._collection = MagicMock()
    vector_store._collection.count.return_value = len(sample_documents)
    vector_store.similarity_search.return_value = sample_documents
    vector_store.similarity_search_with_score.return_value = [
        (doc, 0.8) for doc in sample_documents
    ]
    vector_store.persist = MagicMock()

    workflow = RAGWorkflow(
        mock_config,
        llm=llm,
        embeddings=embeddings,
        vector_store=vector_store,
    )

    workflow.llm = llm
    workflow.vector_store = vector_store
    return workflow


def test_multi_retriever_fanout_includes_graph_when_enabled(
    mock_workflow: RAGWorkflow, sample_documents: list[Document]
) -> None:
    mock_workflow.config.enable_graph_retriever = True

    state: RAGState = {
        "question": "What is enterprise architecture?",
        "normalized_question": "What is enterprise architecture?",
        "status_messages": [],
        "retriever_results": {},
        "retriever_weights": {"semantic": 1.0},
        "errors": [],
    }

    result = mock_workflow.multi_retriever_fanout(state)

    assert "semantic" in result["retriever_results"]
    assert "graph" in result["retriever_results"]
    assert len(result["retriever_results"]["graph"]) == len(sample_documents)
    assert any("graph" in message.lower() for message in result["status_messages"])


def test_workflow_executes_end_to_end(mock_workflow: RAGWorkflow) -> None:
    mock_workflow.llm.invoke.return_value = MagicMock(content="Mock answer [1]")

    result = mock_workflow.run(
        "How do microservices improve agility?",
        retriever_weights={"semantic": 1.0},
    )

    assert result["answer"] == "Mock answer [1]"
    assert result["citations"]
    assert result["documents"]
    assert result["status_messages"][-1] == "✅ Response ready for presentation"


def test_workflow_fallback_when_no_documents(mock_config: RAGConfig) -> None:
    llm = MagicMock()
    embeddings = MagicMock()
    vector_store = MagicMock()

    vector_store._collection = MagicMock()
    vector_store._collection.count.return_value = 0
    vector_store.similarity_search.return_value = []
    vector_store.similarity_search_with_score.return_value = []
    vector_store.add_documents = MagicMock()
    vector_store.persist = MagicMock()

    workflow = RAGWorkflow(
        mock_config,
        llm=llm,
        embeddings=embeddings,
        vector_store=vector_store,
    )

    result = workflow.run("Explain zero trust security.")

    assert "could not locate" in result["answer"].lower()
    assert result["citations"] == []
    assert result["fallback_reason"] == "no_documents"
    llm.invoke.assert_not_called()


def test_ingest_documents_splits_and_persists(
    mock_workflow: RAGWorkflow, sample_documents: list[Document]
) -> None:
    mock_workflow.vector_store.add_documents = MagicMock()
    mock_workflow.vector_store.persist = MagicMock()

    with patch("rag_workflow.RecursiveCharacterTextSplitter") as mock_splitter_cls:
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = sample_documents
        mock_splitter_cls.return_value = mock_splitter

        ingested = mock_workflow.ingest_documents(sample_documents)

    assert ingested == len(sample_documents)
    mock_workflow.vector_store.add_documents.assert_called_once_with(sample_documents)
    mock_workflow.vector_store.persist.assert_called_once()


def test_refresh_index_invokes_persist(mock_workflow: RAGWorkflow) -> None:
    mock_workflow.vector_store.persist = MagicMock()

    assert mock_workflow.refresh_index() is True
    mock_workflow.vector_store.persist.assert_called_once()


def test_describe_graph_contains_expected_nodes(mock_workflow: RAGWorkflow) -> None:
    description = mock_workflow.describe_graph()

    node_names = {node["name"] for node in description["nodes"]}
    expected_nodes = {
        "intake_query",
        "ensure_ingestion_ready",
        "multi_retriever_fanout",
        "fuse_and_rank",
        "generate_context",
        "generate_answer",
        "format_response",
        "handle_no_results",
    }

    assert expected_nodes.issubset(node_names)
    assert any(edge["condition"] == "fallback" for edge in description["edges"])
