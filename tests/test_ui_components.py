import importlib
from unittest.mock import MagicMock, patch

import pytest


def test_core_modules_importable() -> None:
    assert importlib.import_module("ai_rag.core.settings")
    assert importlib.import_module("ai_rag.orchestration.rag_workflow")
    assert importlib.import_module("ai_rag.ui.app")


def test_settings_defaults() -> None:
    from ai_rag.core.settings import Settings

    settings = Settings(google_api_key="key")

    assert settings.environment == "local"
    assert settings.retriever_top_k == 5
    assert settings.embedding_model


@patch("streamlit.set_page_config")
@patch("streamlit.session_state", new_callable=MagicMock)
def test_streamlit_app_initializes(mock_session_state: MagicMock, mock_set_page_config: MagicMock) -> None:
    fake_streamlit = MagicMock()
    fake_streamlit.session_state = mock_session_state
    fake_streamlit.set_page_config = mock_set_page_config

    with patch.dict("sys.modules", {"streamlit": fake_streamlit}):
        module = importlib.import_module("ai_rag.ui.app")
        assert hasattr(module, "initialize_session_state")


def test_rag_state_structure() -> None:
    from ai_rag.orchestration.rag_workflow import RAGState

    state: RAGState = {
        "question": "Test question",
        "documents": [],
        "context": "",
        "answer": "",
        "citations": [],
        "status_messages": [],
        "retriever_weights": {"semantic": 1.0},
    }

    assert "question" in state
    assert "documents" in state
    assert "answer" in state
    assert "citations" in state
