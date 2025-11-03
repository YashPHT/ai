# Enterprise RAG Q&A System with Streamlit

A production-ready Retrieval-Augmented Generation (RAG) system with a Streamlit web interface, powered by LangGraph and Google's Gemini AI.

## Features

- 🤖 **Intelligent Q&A**: Ask questions and get contextual answers with citations
- 📚 **Source Attribution**: View citations and sources for each answer
- ⚙️ **Configurable Retrieval**: Adjust retriever weights and document count
- 📥 **Document Ingestion**: Upload and index new documents via the UI
- 🔄 **Real-time Streaming**: See intermediate processing steps
- 🔐 **Authentication Support**: Optional authentication hook for secure access
- 🛠️ **Admin Controls**: Manage index refresh and ingestion from the UI
- 🐳 **Container Ready**: Docker and Docker Compose support

## Architecture

The system is built with:

- **Streamlit**: Web UI framework
- **LangGraph**: Workflow orchestration for RAG pipeline
- **LangChain**: Document processing and retrieval
- **Google Gemini**: Large language model for answer generation
- **ChromaDB**: Vector database for semantic search
- **Sentence Transformers**: Text embeddings

## Quick Start

### Prerequisites

- Python 3.11+
- Google API Key (for Gemini)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Running Locally

#### Option 1: Direct Python

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

#### Option 2: Custom Port

```bash
streamlit run app.py --server.port 8080
```

#### Option 3: Docker

```bash
docker build -t streamlit-rag .
docker run -p 8501:8501 --env-file .env streamlit-rag
```

#### Option 4: Docker Compose

```bash
docker-compose up
```

## Configuration

### Environment Variables

Configure the application using environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google API key for Gemini | Required |
| `ENVIRONMENT` | Deployment environment (local/production) | `local` |
| `ENABLE_AUTH` | Enable authentication | `false` |
| `RETRIEVER_TOP_K` | Number of documents to retrieve | `5` |
| `CHUNK_SIZE` | Document chunk size | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `VECTOR_STORE_PATH` | Path to ChromaDB storage | `./chroma_db` |

### Runtime Configuration

The UI provides controls to adjust:

- **Semantic Search Weight**: Scale the retrieval weight (0.0 - 2.0)
- **Top K Documents**: Number of documents to retrieve (1 - 20)

## Usage

### Asking Questions

1. Enter your question in the text area
2. Click "Submit" or use the example button
3. View the processing status in real-time
4. See the answer with citations below

### Document Ingestion

1. Open the sidebar admin controls
2. Upload a text file
3. Click "Ingest Document"
4. The document will be chunked and indexed

### Index Management

- **Refresh Index**: Click the "Refresh Index" button to persist changes
- **Clear Conversation**: Remove conversation history

## Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run with hot reload
streamlit run app.py
```

### Production Deployment

#### Using Docker

1. Build the image:
```bash
docker build -t streamlit-rag:latest .
```

2. Run the container:
```bash
docker run -d \
  -p 8501:8501 \
  -e GOOGLE_API_KEY=your_key \
  -e ENVIRONMENT=production \
  -v $(pwd)/chroma_db:/app/chroma_db \
  --name rag-app \
  streamlit-rag:latest
```

#### Using Docker Compose

1. Set environment variables in `.env`

2. Deploy:
```bash
docker-compose up -d
```

3. View logs:
```bash
docker-compose logs -f
```

#### Cloud Deployment Options

**Streamlit Cloud:**
```bash
# Push to GitHub and connect via Streamlit Cloud dashboard
# Add secrets in the Streamlit Cloud UI
```

**AWS/GCP/Azure:**
```bash
# Use Docker deployment method
# Configure load balancer and auto-scaling as needed
```

**Kubernetes:**
```yaml
# See deployment.yaml example (create separately)
kubectl apply -f deployment.yaml
```

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Suite

```bash
# Configuration tests
pytest tests/test_config.py -v

# Workflow tests
pytest tests/test_rag_workflow.py -v

# UI component tests
pytest tests/test_ui_components.py -v
```

### Coverage Report

```bash
pytest --cov=. --cov-report=html tests/
```

## API Reference

### RAGWorkflow

Main workflow class for RAG operations.

```python
from config import RAGConfig
from rag_workflow import RAGWorkflow

config = RAGConfig.from_env()
workflow = RAGWorkflow(config)

# Run query
result = workflow.run("What is microservices architecture?")

# Stream results
for state in workflow.stream("What is cloud computing?"):
    print(state)

# Ingest documents
from langchain.schema import Document
docs = [Document(page_content="...", metadata={"source": "..."})]
num_chunks = workflow.ingest_documents(docs)
```

### Configuration

```python
from config import RAGConfig

# Load from environment
config = RAGConfig.from_env()

# Create manually
config = RAGConfig(
    google_api_key="your_key",
    environment="production",
    retriever_top_k=10
)

# Validate
is_valid, error = config.validate()
```

## Troubleshooting

### Common Issues

**Issue**: `GOOGLE_API_KEY is required`
- **Solution**: Set `GOOGLE_API_KEY` in `.env` file

**Issue**: Vector store initialization fails
- **Solution**: Ensure write permissions for `chroma_db` directory
- **Solution**: Delete `chroma_db` and restart to reinitialize

**Issue**: Import errors
- **Solution**: Install all dependencies: `pip install -r requirements.txt`

**Issue**: Authentication loop
- **Solution**: Use credentials: username=`admin`, password=`admin` (for demo)

### Debug Mode

Enable debug logging:

```bash
export STREAMLIT_LOG_LEVEL=debug
streamlit run app.py
```

## Architecture Details

### RAG Pipeline Flow

```
User Query
    ↓
┌─────────────────────┐
│  Retrieve Documents │ ← Vector Store (ChromaDB)
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Generate Context   │ ← Extract & Format
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Generate Answer    │ ← Gemini LLM
└─────────────────────┘
    ↓
Answer + Citations
```

### Components

- **app.py**: Streamlit UI and user interaction
- **rag_workflow.py**: LangGraph workflow implementation
- **config.py**: Configuration management
- **tests/**: Unit and integration tests

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- Create an issue in the repository
- Check existing documentation
- Review troubleshooting section

## Roadmap

- [ ] Multi-user authentication with database
- [ ] Advanced retriever options (hybrid search, reranking)
- [ ] Conversation memory and history persistence
- [ ] PDF and document format support
- [ ] Batch ingestion from cloud storage
- [ ] Custom model configuration UI
- [ ] Analytics and usage metrics
- [ ] Export/import knowledge base

## Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/)
- [LangChain](https://langchain.com/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Google Generative AI](https://ai.google.dev/)
- [ChromaDB](https://www.trychroma.com/)
