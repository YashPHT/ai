# Enterprise RAG Platform

Enterprise-grade Retrieval Augmented Generation (RAG) system for knowledge-heavy organizations.

This repository provides the foundational architecture, configuration, and tooling to build a production-ready RAG platform. It includes ingestion pipelines, configurable retrieval strategies, ranking/reranking components, orchestration workflows, and a Streamlit UI for experimentation.

## Architecture Overview

```
                          ┌─────────────────────────────────┐
                          │        ai_rag.core              │
                          │  settings • logging • tracing   │
                          └─────────────────────────────────┘
                                       │
                                       ▼
┌───────────────────────┐   ┌───────────────────────┐   ┌─────────────────────────┐
│  ai_rag.ingestion     │   │  ai_rag.retrieval     │   │   ai_rag.ranking        │
│  pdf • web • structured│  │  vector • graph • sw  │   │   rerankers • fusion    │
└───────────────────────┘   └───────────────────────┘   └─────────────────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────────────┐
                          │   ai_rag.orchestration          │
                          │  workflows • state management   │
                          └─────────────────────────────────┘
                                       │
                                       ▼
                               ai_rag.ui (Streamlit)
```

The system is designed to be modular:

- **Ingestion**: Pipelines for PDFs, web pages, and structured data that normalize and persist documents.
- **Retrieval**: Hybrid retrieval strategies (vector, graph-based, sentence windows, Pinecone).
- **Ranking**: Fusion, scoring, and reranking utilities to produce high-quality context.
- **Orchestration**: LangGraph-driven workflows for coordinating ingestion, retrieval, and generation.
- **UI**: Streamlit front-end for iterative experimentation, demoing, and operator workflows.

## Repository Layout

```
ai_rag/
  core/              # Settings, logging, tracing, constants
  ingestion/         # Document loaders and chunking utilities
  retrieval/         # Vector, graph, and hybrid retrieval implementations
  ranking/           # Fusion, rerankers, scoring helpers
  orchestration/     # Workflow coordination and state management
  ui/                # Streamlit application entrypoints
scripts/             # CLI tools, demos
tests/               # Unit and integration tests
```

## Getting Started

### Prerequisites

- Python 3.11+
- Poetry (recommended) or pip/uv
- Access to Gemini and Pinecone (optional for local dev, required for full pipeline)

### Installation

#### Using Poetry

```bash
poetry install
poetry shell
```

#### Using pip

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

Copy the example file and populate it with your secrets:

```bash
cp .env.example .env
```

| Variable | Description | Required |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Google API key for Gemini (alias of `GEMINI_API_KEY`) | ✅ |
| `GEMINI_API_KEY` | Gemini API key (used when `GOOGLE_API_KEY` is not set) | ✅ |
| `PINECONE_API_KEY` | Pinecone API key | Optional |
| `PINECONE_ENVIRONMENT` | Pinecone environment | Optional |
| `PINECONE_INDEX_NAME` | Pinecone index name | Optional |
| `PINECONE_NAMESPACE` | Pinecone namespace | Optional |
| `PINECONE_DIMENSION` | Embedding dimension for Pinecone | Optional |
| `PINECONE_METRIC` | Similarity metric (cosine, dotproduct, euclidean) | Optional |
| `PINECONE_TOP_K` | Number of embeddings to retrieve from Pinecone | Optional |
| `PINECONE_BATCH_SIZE` | Batch size for Pinecone upserts | Optional |
| `RETRIEVER_TOP_K` | Number of documents to retrieve in hybrid retrievers | Optional |
| `CHUNK_SIZE` | Document chunk size | Optional |
| `CHUNK_OVERLAP` | Overlap between chunks | Optional |
| `VECTOR_STORE_PATH` | Local persistence for ChromaDB | Optional |
| `ENABLE_AUTH` | Enable authentication for the UI | Optional |
| `ENABLE_PINECONE_RETRIEVER` | Toggle the Pinecone retriever | Optional |
| `ENABLE_SENTENCE_WINDOW_RETRIEVER` | Toggle sentence window retriever | Optional |
| `ENABLE_GRAPH_RETRIEVER` | Toggle graph retriever | Optional |
| `ENABLE_TELEMETRY` | Enable telemetry/tracing hooks | Optional |
| `ENVIRONMENT` | Runtime environment (`local`, `staging`, `prod`) | Optional |
| `LOG_LEVEL` | Log verbosity (DEBUG, INFO, WARNING, ERROR) | Optional |

The settings module (`ai_rag.core.settings`) uses `pydantic-settings` to load and validate these values at startup.

### Development Workflow

Install dev dependencies, configure git hooks, and initialize logging directories:

```bash
make setup
```

Available commands:

- `make run` – Launch the Streamlit UI.
- `make test` – Execute the test suite with coverage.
- `make lint` – Run Ruff for linting.
- `make format` – Format with Black and sort imports with Ruff.
- `make docker-build` / `make docker-run` – Container workflow.

### Logging & Tracing

- Logging is configured via `ai_rag.core.logging_config.configure_logging`.
- Structured output includes request identifiers and persists to `logs/application.log`.
- Use `ai_rag.core.tracing.trace_execution` decorator or `TracingContext` context manager to trace runtime hotspots.

### Running the App

```bash
make run
```

The Streamlit UI will be available at `http://localhost:8501`.

### Testing

```bash
make test
```

or

```bash
pytest tests/ -v
```

## Pre-commit Hooks

Install the hooks to ensure linting and formatting rules are enforced:

```bash
pre-commit install
pre-commit run --all-files
```

Configuration lives in `.pre-commit-config.yaml` and runs `black`, `ruff`, and several safety checks.

## Extending the Platform

- Add new ingestion pipelines under `ai_rag.ingestion` and register them with the orchestration layer.
- Implement custom retrievers or rerankers in `ai_rag.retrieval` and `ai_rag.ranking`.
- Extend workflows in `ai_rag.orchestration` and update the UI components accordingly.
- Inject custom telemetry or observability by extending `ai_rag.core.tracing`.

## License

MIT License – see [LICENSE](LICENSE) for details.
