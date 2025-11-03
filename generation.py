
import logging
from typing import Any, Dict, List

from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from config import RAGConfig


class GeminiRAG:
    """Handles the generation phase of the RAG pipeline using Google Gemini."""

    def __init__(self, config: RAGConfig, llm: ChatGoogleGenerativeAI):
        self.config = config
        self.llm = llm
        self.logger = logging.getLogger(self.__class__.__name__)

    def _format_context(self, documents: List[Document]) -> str:
        """Formats the retrieved documents into a single context string."""
        context = []
        for doc in documents:
            meta_info = " ".join(
                f"{key}: {value}" for key, value in doc.metadata.items()
            )
            context.append(f"Source: {meta_info}\n\n{doc.page_content}")
        return "\n\n---\n\n".join(context)

    def generate_answer(
        self, question: str, documents: List[Document]
    ) -> Dict[str, Any]:
        """Generates an answer using the provided context and question."""
        context_str = self._format_context(documents)
        prompt_template = ChatPromptTemplate.from_template(self.config.generation_prompt)
        prompt = prompt_template.format(context=context_str, question=question)

        try:
            response = self.llm.invoke(prompt)
            return {
                "answer": response.content,
                "citations": self.extract_citations(response.content, documents),
            }
        except Exception as e:
            self.logger.error(f"Error during Gemini API call: {e}")
            return {"answer": "Error generating answer.", "citations": []}

    def extract_citations(
        self, answer: str, documents: List[Document]
    ) -> List[Dict[str, Any]]:
        """Extracts citations from the generated answer."""
        # This is a placeholder for a more sophisticated citation extraction logic.
        # For now, we'll just return all the sources that were used in the context.
        return [doc.metadata for doc in documents]
