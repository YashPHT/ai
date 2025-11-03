#!/usr/bin/env python3
"""
Smoke test script to verify the RAG application is working correctly.
Run this after installation to ensure everything is set up properly.
"""

import sys
import os
from pathlib import Path

def test_imports():
    print("🔍 Testing imports...")
    try:
        import streamlit
        import langchain
        import langchain_google_genai
        import langgraph
        import chromadb
        from config import RAGConfig
        from rag_workflow import RAGWorkflow
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Run: pip install -r requirements.txt")
        return False


def test_config():
    print("\n🔍 Testing configuration...")
    try:
        from config import RAGConfig
        
        os.environ["GOOGLE_API_KEY"] = "test_key_for_validation"
        config = RAGConfig.from_env()
        
        if not config.google_api_key:
            print("❌ API key not loaded")
            return False
        
        is_valid, error = config.validate()
        if not is_valid and "GOOGLE_API_KEY" not in str(error):
            print(f"❌ Config validation error: {error}")
            return False
        
        print("✅ Configuration system working")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_env_file():
    print("\n🔍 Testing environment file...")
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_example.exists():
        print("❌ .env.example not found")
        return False
    
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("Create one from .env.example and add your GOOGLE_API_KEY")
        return False
    
    with open(env_file) as f:
        content = f.read()
        if "GOOGLE_API_KEY" not in content:
            print("⚠️  GOOGLE_API_KEY not in .env")
            return False
        
        if "your_google_api_key_here" in content:
            print("⚠️  Please replace placeholder API key in .env")
            return False
    
    print("✅ Environment file configured")
    return True


def test_workflow_structure():
    print("\n🔍 Testing workflow structure...")
    try:
        from rag_workflow import RAGWorkflow, RAGState
        from config import RAGConfig
        
        state: RAGState = {
            "question": "test",
            "documents": [],
            "context": "",
            "answer": "",
            "citations": [],
            "status_messages": [],
            "retriever_weights": {"semantic": 1.0}
        }
        
        if not all(key in state for key in ["question", "answer", "citations"]):
            print("❌ RAGState structure invalid")
            return False
        
        print("✅ Workflow structure valid")
        return True
    except Exception as e:
        print(f"❌ Workflow structure error: {e}")
        return False


def test_file_structure():
    print("\n🔍 Testing file structure...")
    required_files = [
        "app.py",
        "config.py",
        "rag_workflow.py",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "README.md",
        "Dockerfile",
        "docker-compose.yml"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files present")
    return True


def test_tests_directory():
    print("\n🔍 Testing test directory...")
    test_dir = Path("tests")
    
    if not test_dir.exists():
        print("❌ tests/ directory not found")
        return False
    
    test_files = list(test_dir.glob("test_*.py"))
    if len(test_files) < 3:
        print(f"⚠️  Only {len(test_files)} test files found")
        return False
    
    print(f"✅ Test directory present with {len(test_files)} test files")
    return True


def main():
    print("=" * 60)
    print("🚀 RAG Application Smoke Test")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Environment File", test_env_file),
        ("Workflow Structure", test_workflow_structure),
        ("Tests Directory", test_tests_directory),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All smoke tests passed! Your application is ready to run.")
        print("\nNext steps:")
        print("1. Ensure GOOGLE_API_KEY is set in .env")
        print("2. Run: streamlit run app.py")
        print("3. Open: http://localhost:8501")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
