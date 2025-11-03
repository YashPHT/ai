.PHONY: help install install-dev setup test lint format clean run docker-build docker-run

help:
    @echo "Enterprise RAG System - Development Commands"
    @echo ""
    @echo "Setup:"
    @echo "  make install        Install production dependencies"
    @echo "  make install-dev    Install development dependencies"
    @echo "  make setup          Full setup: install + pre-commit hooks"
    @echo ""
    @echo "Development:"
    @echo "  make run            Run the Streamlit application"
    @echo "  make test           Run tests with pytest"
    @echo "  make lint           Run linting with ruff"
    @echo "  make format         Format code with black and ruff"
    @echo "  make clean          Clean build artifacts and caches"
    @echo ""
    @echo "Docker:"
    @echo "  make docker-build   Build Docker image"
    @echo "  make docker-run     Run Docker container"

install:
    pip install -r requirements.txt

install-dev:
    pip install -r requirements.txt
    pip install ruff black pre-commit pytest pytest-cov pytest-mock pydantic-settings pinecone-client

setup: install-dev
    pre-commit install
    @echo "Setup complete! Run 'cp .env.example .env' and configure your environment variables."

test:
    pytest tests/ -v --cov=ai_rag --cov-report=term-missing

lint:
    ruff check .

format:
    black .
    ruff check --fix .

clean:
    rm -rf __pycache__ .pytest_cache .coverage htmlcov
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type f -name "*.egg-info" -exec rm -rf {} +

run:
    python -m streamlit run ai_rag/ui/app.py

docker-build:
    docker build -t enterprise-rag:latest .

docker-run:
    docker run -p 8501:8501 --env-file .env enterprise-rag:latest
