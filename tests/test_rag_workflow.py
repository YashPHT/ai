import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from ai_rag.core.settings import Settings
from ai_rag.ranking.fusion import FusionPipeline, KeywordOverlapReranker
from ai_rag.orchestration.rag_workflow import RAGWorkflow, RAGState


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(page_content="Test content 1", metadata={"source": "test1.pdf", "page": 1}),
        Document(page_content="Test content 2", metadata={"source": "test2.pdf", "page": 2}),
    ]


@pytest.fixture
def mock_config() -> Settings:
    return Settings(
        google_api_key="test_key",
        environment="test",
        retriever_top_k=2,
        graph_retriever_top_k=2,
    )


@pytest.fixture
def mock_workflow(mock_config: Settings, sample_documents: list[Document]) -> RAGWorkflow:
    llm = MagicMock()
    embeddings = MagicMock()
    vector_store = MagicMock()

    vector_store._collection = MagicMock()
    vector_store._collection.count.return_value = len(sample_documents)
    vector_store.similarity_search.return_value = sample_documents
    vector_store.similarity_search_with_score.return_value = [
        (doc, 0.8) for doc in sample_documents
    ]
    vector_store.add_documents = MagicMock()
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
    mock_workflow.graph_retriever.reset()
    mock_workflow.graph_retriever.index_documents(sample_documents)
    mock_workflow._initialize_retrievers()

    state: RAGState = {
        "question": "What is enterprise architecture?",
        "normalized_question": "What is enterprise architecture?",
        "status_messages": [],
        "retriever_results": {},
        "retriever_weights": {},
        "errors": [],
    }

    result = mock_workflow.multi_retriever_fanout(state)

    assert "semantic" in result["retriever_results"]
    semantic_results = result["retriever_results"]["semantic"]
    assert all(doc.metadata.get("retriever") == "semantic" for doc in semantic_results)

    assert "graph" in result["retriever_results"]
    graph_results = result["retriever_results"]["graph"]
    assert graph_results  # graph retriever returns contextual sentences
    assert all(doc.metadata.get("retriever") == "graph" for doc in graph_results)
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
    assert result["status_messages"][-1] == "[SUCCESS] Response ready for presentation"

    diagnostics = result["fusion_diagnostics"]
    assert diagnostics["total_candidates"] >= len(result["documents"])
    assert len(diagnostics["selected"]) == len(result["documents"])
    assert diagnostics["token_usage"] >= 0


def test_workflow_fallback_when_no_documents(mock_config: Settings) -> None:
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
    mock_workflow.pinecone_pipeline = MagicMock()
    mock_workflow.pinecone_index_manager = MagicMock()
    mock_workflow.sentence_window_retriever = MagicMock()
    mock_workflow.graph_retriever = MagicMock()

    with patch("ai_rag.orchestration.rag_workflow.RecursiveCharacterTextSplitter") as mock_splitter_cls:
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = sample_documents
        mock_splitter_cls.return_value = mock_splitter

        ingested = mock_workflow.ingest_documents(sample_documents)

    assert ingested == len(sample_documents)
    mock_workflow.vector_store.add_documents.assert_called_once_with(sample_documents)
    mock_workflow.vector_store.persist.assert_called_once()
    mock_workflow.pinecone_index_manager.ensure_index.assert_called_once()
    assert (
        mock_workflow.pinecone_index_manager.ensure_index.call_args.kwargs["metric"]
        == mock_workflow.config.pinecone_metric
    )
    mock_workflow.pinecone_pipeline.upsert_documents.assert_called_once_with(sample_documents)
    mock_workflow.sentence_window_retriever.index_documents.assert_called_once_with(sample_documents)
    mock_workflow.graph_retriever.index_documents.assert_called_once_with(sample_documents)


def test_refresh_index_invokes_persist(mock_workflow: RAGWorkflow) -> None:
    mock_workflow.vector_store.persist = MagicMock()

    assert mock_workflow.refresh_index() is True
    mock_workflow.vector_store.persist.assert_called_once()


def test_fusion_token_budget_diagnostics(
    mock_workflow: RAGWorkflow, sample_documents: list[Document]
) -> None:
    mock_workflow.config.fusion_token_budget = 5
    reranker = (
        KeywordOverlapReranker() if mock_workflow.config.enable_fusion_reranker else None
    )
    mock_workflow.fusion_pipeline = FusionPipeline(
        token_budget=mock_workflow.config.fusion_token_budget,
        max_results=mock_workflow.config.retriever_top_k,
        rrf_k=mock_workflow.config.fusion_rrf_k,
        reranker=reranker,
        reranker_weight=mock_workflow.config.fusion_reranker_weight,
        logger=mock_workflow.logger,
    )

    state: RAGState = {
        "question": "What is enterprise architecture?",
        "normalized_question": "What is enterprise architecture?",
        "status_messages": [],
        "retriever_results": {"semantic": sample_documents},
        "retriever_weights": {"semantic": 1.0},
    }

    updated = mock_workflow.fuse_and_rank(state)

    assert len(updated["fused_documents"]) == 1
    diagnostics = updated["fusion_diagnostics"]
    assert diagnostics["omitted"]
    assert diagnostics["token_usage"] <= mock_workflow.config.fusion_token_budget
    assert any("Context token usage" in message for message in updated["status_messages"])
    assert diagnostics["omitted"][0]["rank"] == 2


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
