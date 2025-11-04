import logging
import os
from operator import add
from typing import Annotated, Any, Callable, Dict, List, Optional, Sequence, TypedDict, Union

from langchain_core.documents import Document
from ai_rag.ingestion import (
    Chunk as IngestionChunk,
    Document as IngestionDocument,
    as_langchain_documents,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, StateGraph

from ai_rag.core.settings import Settings
from ai_rag.ranking.fusion import FusionPipeline, FusionReranker, KeywordOverlapReranker
from ai_rag.orchestration.generation import GeminiRAG
from ai_rag.retrieval import (
    GraphRetriever,
    PineconeEmbeddingPipeline,
    PineconeIndexManager,
    PineconeRetriever,
    SentenceWindowRetriever,
)


class RAGState(TypedDict, total=False):
    """Typed state container flowing through the LangGraph workflow."""

    question: str
    normalized_question: str
    documents: List[Document]
    fused_documents: List[Document]
    context: str
    answer: str
    citations: List[Dict[str, Any]]
    status_messages: Annotated[List[str], add]
    retriever_weights: Dict[str, float]
    retriever_results: Dict[str, List[Document]]
    fusion_diagnostics: Dict[str, Any]
    errors: Annotated[List[str], add]
    active_retrievers: List[str]
    fallback_reason: Optional[str]


IngestibleDocument = Union[Document, IngestionDocument, IngestionChunk]


class RAGWorkflow:
    """End-to-end RAG workflow orchestrated by LangGraph."""

    def __init__(
        self,
        config: Settings,
        *,
        llm: Optional[ChatGoogleGenerativeAI] = None,
        embeddings: Optional[GoogleGenerativeAIEmbeddings] = None,
        vector_store: Optional[Any] = None,
        fusion_reranker: Optional[FusionReranker] = None,
        pinecone_manager: Optional[PineconeIndexManager] = None,
        pinecone_pipeline: Optional[PineconeEmbeddingPipeline] = None,
        pinecone_retriever: Optional[PineconeRetriever] = None,
        sentence_window_retriever: Optional[SentenceWindowRetriever] = None,
        graph_retriever: Optional[GraphRetriever] = None,
    ) -> None:
        self.config = config
        os.environ["GOOGLE_API_KEY"] = config.google_api_key

        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.embeddings = embeddings or GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = llm or ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        self.gemini_rag = GeminiRAG(config=self.config, llm=self.llm)

        self.pinecone_index_manager = pinecone_manager
        self.pinecone_pipeline = pinecone_pipeline
        self.pinecone_retriever = pinecone_retriever
        self.sentence_window_retriever = sentence_window_retriever
        self.graph_retriever = graph_retriever
        self._retriever_dispatch: Dict[str, Callable[[str, float], List[Document]]] = {}
        self._retriever_order: List[str] = []
        self._event_handlers: List[Callable[[str, Dict[str, Any]], None]] = []
        
        self._mock_vector_store = vector_store

        if vector_store is None and pinecone_retriever is None:
            self._initialize_vector_store()
        elif vector_store is not None:
            self.logger.info("Using provided mock vector store (backwards compatibility)")

        reranker_instance = fusion_reranker
        if reranker_instance is None and self.config.enable_fusion_reranker:
            reranker_instance = KeywordOverlapReranker()

        self.fusion_pipeline = FusionPipeline(
            token_budget=self.config.fusion_token_budget,
            max_results=self.config.retriever_top_k,
            rrf_k=self.config.fusion_rrf_k,
            reranker=reranker_instance,
            reranker_weight=self.config.fusion_reranker_weight,
            logger=self.logger,
        )

        self._initialize_retrievers()
        self._ensure_vector_store_seeded()

        self._graph_nodes: List[str] = []
        self._graph_edges: List[Dict[str, Optional[str]]] = []
        self.workflow = self._build_workflow()

    def register_event_handler(self, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        """Register a callback that receives telemetry events."""

        self._event_handlers.append(handler)

    def _emit_event(self, name: str, payload: Dict[str, Any]) -> None:
        if not self.config.enable_telemetry:
            return

        for handler in list(self._event_handlers):
            try:
                handler(name, payload)
            except Exception as exc:  # pragma: no cover - defensive logging only
                self.logger.warning("Telemetry handler error: %s", exc)

    def _initialize_vector_store(self) -> None:
        """Initialize Pinecone as the primary vector store."""
        try:
            if self.pinecone_index_manager is None:
                self.pinecone_index_manager = PineconeIndexManager(
                    index_name=self.config.pinecone_index_name,
                    dimension=self.config.pinecone_dimension,
                    api_key=self.config.pinecone_api_key or None,
                    environment=self.config.pinecone_environment or None,
                    namespace=self.config.pinecone_namespace,
                    logger=self.logger,
                )
            if self.pinecone_pipeline is None:
                self.pinecone_pipeline = PineconeEmbeddingPipeline(
                    index_manager=self.pinecone_index_manager,
                    embeddings=self.embeddings,
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                    batch_size=self.config.pinecone_batch_size,
                    namespace=self.config.pinecone_namespace,
                    logger=self.logger,
                )
            if self.pinecone_retriever is None:
                self.pinecone_retriever = PineconeRetriever(
                    index_manager=self.pinecone_index_manager,
                    embeddings=self.embeddings,
                    top_k=self.config.pinecone_top_k,
                    namespace=self.config.pinecone_namespace,
                )
            self.logger.info("Pinecone vector store initialized successfully")
        except Exception as error:
            self.logger.warning("Failed to initialize Pinecone vector store: %s", error)
            self.pinecone_index_manager = None
            self.pinecone_pipeline = None
            self.pinecone_retriever = None

    def _register_retriever(
        self, name: str, handler: Callable[[str, float], List[Document]]
    ) -> None:
        self._retriever_dispatch[name] = handler
        if name not in self._retriever_order:
            self._retriever_order.append(name)

    def _initialize_retrievers(self) -> None:
        self._retriever_dispatch.clear()
        self._retriever_order.clear()

        self._register_retriever("semantic", self._semantic_retrieval)

        if self.config.enable_pinecone_retriever and self.pinecone_retriever:
            self._register_retriever("pinecone", self._pinecone_retrieval)

        if self.sentence_window_retriever is None:
            self.sentence_window_retriever = SentenceWindowRetriever(
                window_size=self.config.sentence_window_size,
                top_k=self.config.sentence_retriever_top_k,
            )
        else:
            self.sentence_window_retriever.window_size = max(0, self.config.sentence_window_size)
            self.sentence_window_retriever.top_k = max(1, self.config.sentence_retriever_top_k)

        if self.config.enable_sentence_window_retriever:
            self._register_retriever("sentence_window", self._sentence_window_retrieval)

        if self.graph_retriever is None:
            self.graph_retriever = GraphRetriever(
                max_depth=self.config.graph_max_depth,
                top_k=self.config.graph_retriever_top_k,
            )
        else:
            self.graph_retriever.max_depth = max(1, self.config.graph_max_depth)
            self.graph_retriever.top_k = max(1, self.config.graph_retriever_top_k)

        if self.config.enable_graph_retriever:
            self._register_retriever("graph", self._graph_retrieval)

    def _load_sample_documents(self) -> None:
        sample_docs = [
            Document(
                page_content=(
                    "Enterprise software architecture involves designing scalable, maintainable, and secure systems. "
                    "Key principles include separation of concerns, modularity, and adherence to design patterns."
                ),
                metadata={"source": "architecture_guide.pdf", "page": 1},
            ),
            Document(
                page_content=(
                    "Microservices architecture breaks down applications into smaller, independent services. "
                    "Each service handles a specific business capability and communicates via APIs."
                ),
                metadata={"source": "microservices_handbook.pdf", "page": 3},
            ),
            Document(
                page_content=(
                    "Cloud computing provides on-demand access to computing resources. "
                    "Major cloud providers include AWS, Azure, and Google Cloud Platform. "
                    "Benefits include scalability, cost-efficiency, and global reach."
                ),
                metadata={"source": "cloud_basics.pdf", "page": 5},
            ),
            Document(
                page_content=(
                    "DevOps practices combine development and operations to improve deployment frequency. "
                    "Key practices include continuous integration, continuous delivery, and infrastructure as code."
                ),
                metadata={"source": "devops_guide.pdf", "page": 2},
            ),
            Document(
                page_content=(
                    "Security best practices for enterprise applications include authentication, authorization, "
                    "encryption, input validation, and regular security audits. Zero-trust architecture is becoming standard."
                ),
                metadata={"source": "security_handbook.pdf", "page": 7},
            ),
        ]

        ingested_chunks = self.ingest_documents(sample_docs)
        self.logger.info("Loaded %s sample documents into the retrieval suite", len(sample_docs))
        self._emit_event(
            "ingestion.sample_loaded",
            {"documents": len(sample_docs), "chunks": ingested_chunks},
        )

    def _get_index_count(self) -> int:
        if self._mock_vector_store is not None:
            collection = getattr(self._mock_vector_store, "_collection", None)
            if collection and hasattr(collection, "count"):
                try:
                    return int(collection.count())
                except Exception:
                    return 0
            return 0
        
        if not self.pinecone_index_manager:
            return 0

        try:
            stats = self.pinecone_index_manager.get_index_stats()
            total_count = stats.get("total_vector_count", 0)
            namespace_stats = stats.get("namespaces", {})
            namespace_count = namespace_stats.get(self.config.pinecone_namespace, {}).get("vector_count", 0)
            return max(total_count, namespace_count)
        except Exception:
            return 0

    def _ensure_vector_store_seeded(self) -> None:
        if self._mock_vector_store is not None:
            return
        
        if not self.pinecone_index_manager:
            return

        if self._get_index_count() == 0:
            self._load_sample_documents()

    def intake_query(self, state: RAGState) -> RAGState:
        question = state.get("question", "").strip()
        normalized_question = " ".join(question.split())

        status_messages = list(state.get("status_messages", []))
        status_messages.append("[INPUT] Received query for processing")

        self.logger.info("Received question: %s", normalized_question)
        self._emit_event("query.intake", {"question": normalized_question})

        return {
            **state,
            "question": question,
            "normalized_question": normalized_question,
            "status_messages": status_messages,
        }

    def ensure_ingestion_ready(self, state: RAGState) -> RAGState:
        status_messages = list(state.get("status_messages", []))
        current_count = self._get_index_count()

        if current_count == 0:
            status_messages.append("[INDEXING] Knowledge index empty - loading sample corpus")
            self._load_sample_documents()
            current_count = self._get_index_count()
        else:
            status_messages.append(f"[INDEXING] Knowledge index ready with {current_count} documents")

        self._emit_event("ingestion.status", {"document_count": current_count})
        return {**state, "status_messages": status_messages}

    def multi_retriever_fanout(self, state: RAGState) -> RAGState:
        status_messages = list(state.get("status_messages", []))
        retriever_weights = dict(state.get("retriever_weights", {}))

        dispatch_order = self._retriever_order or ["semantic"]
        for name in dispatch_order:
            retriever_weights.setdefault(name, 1.0)

        active_retrievers = [name for name in dispatch_order if retriever_weights.get(name, 0.0) > 0]
        if not active_retrievers:
            active_retrievers = dispatch_order[:1]

        status_messages.append(
            f"[RETRIEVAL] Fan-out to retrievers: {', '.join(active_retrievers)}"
        )

        retriever_results = dict(state.get("retriever_results", {}))
        errors = list(state.get("errors", []))

        for retriever_name in active_retrievers:
            try:
                documents = self._run_retriever(retriever_name, state, retriever_weights)
                retriever_results[retriever_name] = documents
                status_messages.append(
                    f"[RETRIEVAL] {retriever_name.replace('_', ' ').title()} retriever returned {len(documents)} documents"
                )
                self._emit_event(
                    "retriever.success",
                    {"name": retriever_name, "count": len(documents)},
                )
            except Exception as error:  # pragma: no cover - defensive branch
                message = f"[ERROR] {retriever_name.replace('_', ' ').title()} retriever failed: {error}"
                status_messages.append(message)
                errors.append(f"{retriever_name}: {error}")
                self.logger.warning(message)
                self._emit_event(
                    "retriever.error",
                    {"name": retriever_name, "error": str(error)},
                )

        return {
            **state,
            "status_messages": status_messages,
            "retriever_results": retriever_results,
            "errors": errors,
            "active_retrievers": active_retrievers,
            "retriever_weights": retriever_weights,
        }

    def _run_retriever(
        self,
        retriever_name: str,
        state: RAGState,
        weights: Dict[str, float],
    ) -> List[Document]:
        question = state.get("normalized_question") or state.get("question", "")
        handler = self._retriever_dispatch.get(retriever_name)
        if not handler:
            raise ValueError(f"Unknown retriever '{retriever_name}'")

        weight = float(weights.get(retriever_name, 1.0))
        if weight <= 0:
            return []

        return handler(question, weight)

    def _semantic_retrieval(self, question: str, weight: float) -> List[Document]:
        if not question.strip():
            return []

        effective_weight = max(weight, 0.0)
        if effective_weight <= 0:
            return []

        top_k = max(1, min(int(self.config.retriever_top_k * max(effective_weight, 0.1)), 50))
        
        if self._mock_vector_store is not None:
            with_score = getattr(self._mock_vector_store, "similarity_search_with_score", None)
            if callable(with_score):
                results = with_score(question, k=top_k)
                documents: List[Document] = []
                for doc, score in results:
                    metadata = dict(doc.metadata or {})
                    try:
                        metadata["score"] = float(score)
                    except (TypeError, ValueError):
                        metadata["score"] = score
                    metadata["retriever"] = "semantic"
                    documents.append(Document(page_content=doc.page_content, metadata=metadata))
                return documents
            
            raw_docs = self._mock_vector_store.similarity_search(question, k=top_k)
            documents = []
            for doc in raw_docs:
                metadata = dict(doc.metadata or {})
                metadata.setdefault("score", 0.0)
                metadata["retriever"] = "semantic"
                documents.append(Document(page_content=doc.page_content, metadata=metadata))
            return documents
        
        if not self.pinecone_retriever:
            raise RuntimeError("Vector store not initialized")
        
        results = self.pinecone_retriever.retrieve(question, k=top_k)
        
        documents = []
        for result in results:
            doc = result.to_document()
            metadata = dict(doc.metadata or {})
            metadata["retriever"] = "semantic"
            documents.append(Document(page_content=doc.page_content, metadata=metadata))
        
        return documents

    def _pinecone_retrieval(self, question: str, weight: float) -> List[Document]:
        if not self.pinecone_retriever:
            raise RuntimeError("Pinecone retriever not configured")
        if not question.strip():
            return []

        effective_weight = max(weight, 0.0)
        if effective_weight <= 0:
            return []

        top_k = max(1, min(int(self.config.pinecone_top_k * max(effective_weight, 0.1)), 50))
        results = self.pinecone_retriever.retrieve(question, k=top_k)
        return [result.to_document() for result in results]

    def _sentence_window_retrieval(self, question: str, weight: float) -> List[Document]:
        if not self.sentence_window_retriever:
            raise RuntimeError("Sentence window retriever not configured")
        if not question.strip():
            return []

        effective_weight = max(weight, 0.0)
        if effective_weight <= 0:
            return []

        top_k = max(1, min(int(self.config.sentence_retriever_top_k * max(effective_weight, 0.1)), 50))
        results = self.sentence_window_retriever.retrieve(question, k=top_k)
        return [result.to_document() for result in results]

    def _graph_retrieval(self, question: str, weight: float) -> List[Document]:
        if not self.graph_retriever:
            raise RuntimeError("Graph retriever not configured")
        if not question.strip():
            return []

        effective_weight = max(weight, 0.0)
        if effective_weight <= 0:
            return []

        top_k = max(1, min(int(self.config.graph_retriever_top_k * max(effective_weight, 0.1)), 50))
        results = self.graph_retriever.retrieve(
            question,
            k=top_k,
            max_depth=self.config.graph_max_depth,
        )
        return [result.to_document() for result in results]

    def fuse_and_rank(self, state: RAGState) -> RAGState:
        status_messages = list(state.get("status_messages", []))
        retriever_results = state.get("retriever_results", {})
        retriever_weights = state.get("retriever_weights", {"semantic": 1.0})
        question = state.get("normalized_question") or state.get("question", "")

        fusion_result = self.fusion_pipeline.fuse(
            question,
            retriever_results,
            retriever_weights,
        )

        fused_documents = fusion_result.documents

        message = (
            f"[FUSION] Fused {fusion_result.total_candidates} documents into "
            f"{len(fused_documents)} ranked results"
        )
        if fusion_result.truncated:
            message += f" (trimmed {len(fusion_result.truncated)} for token budget)"
        status_messages.append(message)

        if fusion_result.metadata.get("reranker_applied"):
            status_messages.append("[FUSION] Fusion reranker applied to refine ordering")

        token_budget = fusion_result.metadata.get("token_budget")
        if token_budget is not None:
            usage_message = f"[FUSION] Context token usage {fusion_result.token_usage}"
            if token_budget:
                usage_message += f"/{token_budget}"
            if fusion_result.truncated:
                usage_message += f" with {len(fusion_result.truncated)} omitted"
            status_messages.append(usage_message)

        diagnostics = {
            **fusion_result.metadata,
            "selected": fusion_result.selected,
            "omitted": fusion_result.truncated,
            "token_usage": fusion_result.token_usage,
            "total_candidates": fusion_result.total_candidates,
            "deduplicated_candidates": fusion_result.deduplicated_candidates,
        }

        self._emit_event(
            "fusion.completed",
            {
                "selected": len(fusion_result.selected),
                "token_usage": fusion_result.token_usage,
                "total_candidates": fusion_result.total_candidates,
                "deduplicated_candidates": fusion_result.deduplicated_candidates,
            },
        )

        return {
            **state,
            "status_messages": status_messages,
            "fused_documents": fused_documents,
            "documents": fused_documents,
            "fusion_diagnostics": diagnostics,
        }

    def _route_post_fusion(self, state: RAGState) -> str:
        fused = state.get("fused_documents", [])
        if fused:
            return "continue"
        return "fallback"

    def handle_no_results(self, state: RAGState) -> RAGState:
        status_messages = list(state.get("status_messages", []))
        message = "[WARNING] No relevant documents found. Returning fallback guidance."
        status_messages.append(message)

        fallback_answer = (
            "I could not locate supporting documents for your question. "
            "Please verify the knowledge base or ingest additional materials."
        )

        self._emit_event("workflow.fallback", {"reason": "no_documents"})
        return {
            **state,
            "status_messages": status_messages,
            "context": "",
            "citations": [],
            "answer": fallback_answer,
            "fallback_reason": "no_documents",
        }

    def generate_answer(self, state: RAGState) -> RAGState:
        question = state.get("question", "")
        documents = state.get("fused_documents")
        if not documents:
            return {**state, "answer": "No documents found to answer the question."}

        status_messages = list(state.get("status_messages", []))
        status_messages.append("[GENERATION] Generating final answer...")

        generation_result = self.gemini_rag.generate_answer(question, documents)

        self._emit_event(
            "generation.answer",
            {
                "question": question,
                "answer": generation_result["answer"],
                "citations": generation_result["citations"],
            },
        )
        return {
            **state,
            "answer": generation_result["answer"],
            "citations": generation_result["citations"],
            "status_messages": status_messages,
        }

    def _should_fallback(self, state: RAGState) -> bool:
        return not state.get("fused_documents")

    def fallback_answer(self, state: RAGState) -> RAGState:
        question = state.get("question", "")
        status_messages = list(state.get("status_messages", []))
        errors = list(state.get("errors", []))
        error_str = "\n".join(errors)
        fallback_reason = "No documents were retrieved to answer the query."

        self.logger.warning("No documents retrieved for: %s", question)
        status_messages.append(f"[WARNING] {fallback_reason}")
        self._emit_event(
            "generation.fallback",
            {"question": question, "errors": error_str, "reason": fallback_reason},
        )

        return {
            **state,
            "answer": "I could not find an answer based on the available information.",
            "citations": [],
            "status_messages": status_messages,
            "fallback_reason": fallback_reason,
        }

    def ingest_documents(
        self, documents: Sequence[IngestibleDocument]
    ) -> int:
        """Ingests a list of documents into the RAG system."""
        if not documents:
            return 0

        self.logger.info("Ingesting %s documents", len(documents))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
        )

        langchain_docs = as_langchain_documents(documents)
        chunks = text_splitter.split_documents(langchain_docs)

        if self.pinecone_pipeline:
            try:
                self.pinecone_pipeline.run(langchain_docs)
            except Exception as error:
                self.logger.warning("Pinecone ingestion failed: %s", error)

        self.logger.info("Ingested %s chunks", len(chunks))
        self._emit_event("ingestion.completed", {"documents": len(documents), "chunks": len(chunks)})

        return len(chunks)

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(RAGState)

        self._graph_nodes = [
            "intake_query",
            "ensure_ingestion_ready",
            "multi_retriever_fanout",
            "fuse_and_rank",
            "generate_answer",
            "handle_no_results",
        ]

        workflow.add_node("intake_query", self.intake_query)
        workflow.add_node("ensure_ingestion_ready", self.ensure_ingestion_ready)
        workflow.add_node("multi_retriever_fanout", self.multi_retriever_fanout)
        workflow.add_node("fuse_and_rank", self.fuse_and_rank)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("handle_no_results", self.handle_no_results)

        workflow.set_entry_point("intake_query")
        workflow.add_edge("intake_query", "ensure_ingestion_ready")
        workflow.add_edge("ensure_ingestion_ready", "multi_retriever_fanout")
        workflow.add_edge("multi_retriever_fanout", "fuse_and_rank")

        workflow.add_conditional_edges(
            "fuse_and_rank",
            self._route_post_fusion,
            {"continue": "generate_answer", "fallback": "handle_no_results"},
        )

        workflow.add_edge("generate_answer", END)
        workflow.add_edge("handle_no_results", END)

        self._graph_edges = [
            {"source": "intake_query", "target": "ensure_ingestion_ready"},
            {"source": "ensure_ingestion_ready", "target": "multi_retriever_fanout"},
            {"source": "multi_retriever_fanout", "target": "fuse_and_rank"},
            {"source": "fuse_and_rank", "target": "generate_answer"},
            {"source": "fuse_and_rank", "target": "handle_no_results"},
            {"source": "generate_answer", "target": "__end__"},
            {"source": "handle_no_results", "target": "__end__"},
        ]

        return workflow.compile()

    def stream(self, query: str, config: Optional[Dict[str, Any]] = None) -> Any:
        self.logger.info("Streaming RAG workflow for query: %s", query)
        config = config or {}
        config.setdefault("recursion_limit", 10)

        return self.workflow.stream(
            {"question": query, "status_messages": [], "errors": []},
            config=config,
        )

    def invoke(self, query: str, config: Optional[Dict[str, Any]] = None) -> RAGState:
        self.logger.info("Invoking RAG workflow for query: %s", query)
        config = config or {}
        config.setdefault("recursion_limit", 10)

        return self.workflow.invoke(
            {"question": query, "status_messages": [], "errors": []},
            config=config,
        )

    def run(
        self, query: str, retriever_weights: Optional[Dict[str, float]] = None
    ) -> RAGState:
        """Execute the RAG workflow with specified retriever weights."""
        config = {"recursion_limit": 10}
        initial_state: RAGState = {
            "question": query,
            "status_messages": [],
            "errors": [],
            "retriever_weights": retriever_weights or {"semantic": 1.0},
        }
        
        result = self.workflow.invoke(initial_state, config=config)
        
        status_messages = list(result.get("status_messages", []))
        status_messages.append("[SUCCESS] Response ready for presentation")
        result["status_messages"] = status_messages
        
        return result

    def refresh_index(self) -> bool:
        """Refresh the Pinecone index statistics."""
        if not self.pinecone_index_manager:
            return False
        
        try:
            self.pinecone_index_manager.get_index_stats()
            return True
        except Exception as error:
            self.logger.warning("Failed to refresh index: %s", error)
            return False

    def describe_graph(self) -> Dict[str, Any]:
        """Return a description of the workflow graph structure."""
        return {
            "nodes": [{"name": node} for node in self._graph_nodes],
            "edges": [
                {
                    "source": edge["source"],
                    "target": edge["target"],
                    "condition": edge.get("condition"),
                }
                for edge in self._graph_edges
            ],
        }
