#!/bin/bash

set -e

echo "🚀 Starting Enterprise RAG Q&A System..."
echo ""

if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env and add your GOOGLE_API_KEY"
    echo ""
fi

if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
    echo ""
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

if [ ! -f "venv/installed" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    touch venv/installed
    echo "✅ Dependencies installed"
    echo ""
fi

source .env 2>/dev/null || true

if [ -z "$GOOGLE_API_KEY" ] || [ "$GOOGLE_API_KEY" = "your_google_api_key_here" ]; then
    echo "⚠️  WARNING: GOOGLE_API_KEY not set or using default value"
    echo "Please set your API key in .env file"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "🌐 Starting Streamlit app..."
echo "📍 App will be available at http://localhost:8501"
echo ""

streamlit run app.py
