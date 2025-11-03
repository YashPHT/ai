# Quick Start Guide

Get the Enterprise RAG Q&A System running in under 5 minutes!

## Prerequisites

- Python 3.11 or higher
- Google API Key ([Get one here](https://makersuite.google.com/app/apikey))

## Installation

### Option 1: Automated Script (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Make the script executable
chmod +x run.sh

# Run the application
./run.sh
```

The script will:
1. Create `.env` from `.env.example` if needed
2. Set up a virtual environment
3. Install dependencies
4. Start the Streamlit app

### Option 2: Manual Setup

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

### Option 3: Docker

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 2. Run with Docker Compose
docker-compose up
```

## First Steps

1. **Open the application** at `http://localhost:8501`

2. **Try an example query**
   - Click the "💡 Try Example" button
   - Or type your own question like:
     - "What are the key principles of microservices architecture?"
     - "Explain cloud computing benefits"
     - "What are DevOps best practices?"

3. **View the results**
   - Answer with citations
   - Source documents
   - Processing status

4. **Explore admin controls** (in sidebar)
   - Adjust retriever weights
   - Upload and ingest documents
   - Refresh the index

## Configuration

### Basic Configuration

Edit `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key_here
ENVIRONMENT=local
ENABLE_AUTH=false
RETRIEVER_TOP_K=5
```

### Advanced Configuration

See [README.md](README.md) for all configuration options.

## Testing the Application

### Test the UI

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_ui_components.py -v
```

### Manual Testing

1. **Ask a question** - Verify answer with citations
2. **Upload a document** - Try ingesting `sample_data.txt`
3. **Adjust settings** - Change retriever weights
4. **Refresh index** - Test the admin control

## Troubleshooting

### "GOOGLE_API_KEY is required"

Make sure your `.env` file has:
```env
GOOGLE_API_KEY=your_actual_api_key
```

### Import Errors

Install dependencies:
```bash
pip install -r requirements.txt
```

### Port Already in Use

Use a different port:
```bash
streamlit run app.py --server.port 8080
```

### Can't Connect to Streamlit

Check if the app is running:
```bash
# Check logs for errors
docker-compose logs -f  # If using Docker
```

## Next Steps

- Read the [full README](README.md) for detailed features
- Check [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment
- See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

## Support

- **Issues**: Create an issue in the repository
- **Documentation**: See README.md and DEPLOYMENT.md
- **Examples**: Try the sample questions in the UI

## Quick Commands Reference

```bash
# Development
streamlit run app.py                    # Run locally
streamlit run app.py --server.port 8080 # Custom port

# Testing
pytest tests/ -v                        # Run all tests
pytest tests/test_config.py -v          # Run specific tests

# Docker
docker-compose up                       # Start with Docker
docker-compose down                     # Stop containers
docker-compose logs -f                  # View logs

# Environment
source venv/bin/activate                # Activate venv (Linux/Mac)
venv\Scripts\activate                   # Activate venv (Windows)
deactivate                              # Deactivate venv
```

## Sample Questions

Try these questions to test the system:

1. "What are the main cloud service models?"
2. "Explain microservices architecture benefits"
3. "What are security best practices for enterprise applications?"
4. "Describe DevOps practices"
5. "What types of databases are used in enterprise systems?"

Happy querying! 🚀
