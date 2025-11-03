import os
import pytest
from config import RAGConfig


def test_config_from_env():
    os.environ["GOOGLE_API_KEY"] = "test_key"
    os.environ["ENVIRONMENT"] = "test"
    os.environ["RETRIEVER_TOP_K"] = "10"
    
    config = RAGConfig.from_env()
    
    assert config.google_api_key == "test_key"
    assert config.environment == "test"
    assert config.retriever_top_k == 10


def test_config_validation_success():
    config = RAGConfig(google_api_key="valid_key")
    is_valid, error = config.validate()
    
    assert is_valid is True
    assert error is None


def test_config_validation_missing_api_key():
    config = RAGConfig(google_api_key="")
    is_valid, error = config.validate()
    
    assert is_valid is False
    assert "GOOGLE_API_KEY" in error


def test_config_validation_invalid_top_k():
    config = RAGConfig(google_api_key="valid_key", retriever_top_k=0)
    is_valid, error = config.validate()
    
    assert is_valid is False
    assert "RETRIEVER_TOP_K" in error
