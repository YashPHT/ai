import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    try:
        import app
        import config
        import rag_workflow
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_config_initialization():
    from config import RAGConfig
    
    config = RAGConfig(
        google_api_key="test_key",
        environment="test"
    )
    
    assert config.google_api_key == "test_key"
    assert config.environment == "test"


@patch('streamlit.set_page_config')
@patch('streamlit.session_state', new_callable=MagicMock)
def test_app_page_config(mock_session_state, mock_set_page_config):
    with patch.dict(sys.modules, {'streamlit': MagicMock()}):
        try:
            import app
            assert True
        except Exception as e:
            pytest.fail(f"App initialization failed: {e}")


def test_authentication_hook_disabled():
    from config import RAGConfig
    
    config = RAGConfig(google_api_key="test_key", enable_auth=False)
    
    assert config.enable_auth is False


def test_authentication_hook_enabled():
    from config import RAGConfig
    
    config = RAGConfig(google_api_key="test_key", enable_auth=True)
    
    assert config.enable_auth is True


def test_rag_state_structure():
    from rag_workflow import RAGState
    
    state: RAGState = {
        "question": "Test question",
        "documents": [],
        "context": "",
        "answer": "",
        "citations": [],
        "status_messages": [],
        "retriever_weights": {"semantic": 1.0}
    }
    
    assert "question" in state
    assert "documents" in state
    assert "answer" in state
    assert "citations" in state
