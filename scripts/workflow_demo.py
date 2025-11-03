#!/usr/bin/env python3
"""Utility script to visualise and exercise the LangGraph RAG workflow.

This script prints the workflow topology and runs the pipeline end-to-end on
sample in-memory documents using mocked LLM and embedding components. It avoids
network calls so it can be executed in constrained environments.
"""

import os
from pprint import pprint
from unittest.mock import MagicMock

from langchain.schema import Document

from ai_rag.core.settings import Settings
from ai_rag.orchestration.rag_workflow import RAGWorkflow


def build_mock_workflow(config: Settings) -> RAGWorkflow:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="This is a synthesized answer referencing [1]."
    )

    mock_embeddings = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_store._collection = MagicMock()
    mock_vector_store._collection.count.return_value = 2

    documents = [
        Document(
            page_content="Sample architecture guidance emphasising modular design.",
            metadata={"source": "sample_architecture.pdf", "page": 1},
        ),
        Document(
            page_content="Microservices improve agility through independent deployments.",
            metadata={"source": "microservices_notes.pdf", "page": 4},
        ),
    ]

    mock_vector_store.similarity_search.return_value = documents
    mock_vector_store.similarity_search_with_score.return_value = [
        (doc, 0.85) for doc in documents
    ]
    mock_vector_store.persist = MagicMock()

    workflow = RAGWorkflow(
        config,
        llm=mock_llm,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    return workflow


def main() -> None:
    os.environ.setdefault("GOOGLE_API_KEY", "dummy-key")

    config = Settings.from_env()
    config.enable_graph_retriever = True
    config.enable_telemetry = False
    config.retriever_top_k = 2
    config.graph_retriever_top_k = 2

    workflow = build_mock_workflow(config)

    print("\nLangGraph Workflow Structure")
    print("-" * 32)

    description = workflow.describe_graph()
    print("Nodes:")
    for node in description["nodes"]:
        print(f"  • {node['name']}")

    print("Edges:")
    for edge in description["edges"]:
        condition = f" [{edge['condition']}]" if edge.get("condition") else ""
        print(f"  • {edge['source']} -> {edge['target']}{condition}")

    print("\nExecuting sample pipeline...\n")

    result = workflow.run(
        "How do microservices enable continuous delivery?",
        retriever_weights={"semantic": 1.0, "graph": 0.8},
    )

    pprint(
        {
            "answer": result.get("answer"),
            "citations": result.get("citations"),
            "status_messages": result.get("status_messages"),
        }
    )


if __name__ == "__main__":
    main()
