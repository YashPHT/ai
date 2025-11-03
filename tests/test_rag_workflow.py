import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document
from config import RAGConfig
from rag_workflow import RAGWorkflow, RAGState


@pytest.fixture
def mock_config():
    return RAGConfig(
        google_api_key="test_key",
        environment="test",
        retriever_top_k=5
    )


@pytest.fixture
def mock_workflow(mock_config):
    with patch('rag_workflow.ChatGoogleGenerativeAI'), \
         patch('rag_workflow.GoogleGenerativeAIEmbeddings'), \
         patch('rag_workflow.Chroma'):
        workflow = RAGWorkflow(mock_config)
        workflow.vector_store = Mock()
        return workflow


def test_retrieve_documents(mock_workflow):
    mock_docs = [
        Document(page_content="Test content 1", metadata={"source": "test1.pdf"}),
        Document(page_content="Test content 2", metadata={"source": "test2.pdf"})
    ]
    
    mock_workflow.vector_store.similarity_search = Mock(return_value=mock_docs)
    
    state: RAGState = {
        "question": "Test question?",
        "documents": [],
        "context": "",
        "answer": "",
        "citations": [],
        "status_messages": [],
        "retriever_weights": {"semantic": 1.0}
    }
    
    result = mock_workflow.retrieve_documents(state)
    
    assert len(result["documents"]) == 2
    assert len(result["status_messages"]) > 0
    assert any("Retrieved" in msg for msg in result["status_messages"])


def test_generate_context(mock_workflow):
    docs = [
        Document(page_content="Content 1", metadata={"source": "test1.pdf", "page": 1}),
        Document(page_content="Content 2", metadata={"source": "test2.pdf", "page": 2})
    ]
    
    state: RAGState = {
        "question": "Test?",
        "documents": docs,
        "context": "",
        "answer": "",
        "citations": [],
        "status_messages": [],
        "retriever_weights": {"semantic": 1.0}
    }
    
    result = mock_workflow.generate_context(state)
    
    assert result["context"] != ""
    assert len(result["citations"]) == 2
    assert result["citations"][0]["source"] == "test1.pdf"


def test_generate_answer(mock_workflow):
    mock_response = Mock()
    mock_response.content = "This is a test answer [1]"
    mock_workflow.llm.invoke = Mock(return_value=mock_response)
    
    state: RAGState = {
        "question": "Test?",
        "documents": [],
        "context": "Test context",
        "answer": "",
        "citations": [],
        "status_messages": [],
        "retriever_weights": {"semantic": 1.0}
    }
    
    result = mock_workflow.generate_answer(state)
    
    assert result["answer"] == "This is a test answer [1]"
    assert any("generated" in msg.lower() for msg in result["status_messages"])


def test_ingest_documents(mock_workflow):
    mock_workflow.vector_store.add_documents = Mock()
    mock_workflow.vector_store.persist = Mock()
    
    docs = [Document(page_content="Test content", metadata={"source": "test.pdf"})]
    
    with patch('rag_workflow.RecursiveCharacterTextSplitter') as mock_splitter:
        mock_splitter_instance = Mock()
        mock_splitter_instance.split_documents = Mock(return_value=docs)
        mock_splitter.return_value = mock_splitter_instance
        
        num_chunks = mock_workflow.ingest_documents(docs)
        
        assert num_chunks == 1
        mock_workflow.vector_store.add_documents.assert_called_once()


def test_refresh_index(mock_workflow):
    mock_workflow.vector_store.persist = Mock()
    
    result = mock_workflow.refresh_index()
    
    assert result is True
    mock_workflow.vector_store.persist.assert_called_once()
