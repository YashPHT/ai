import logging
import os
from operator import add
from typing import Annotated, Any, Callable, Dict, List, Optional, Sequence, TypedDict

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import END, StateGraph

from config import RAGConfig
from fusion import FusionPipeline, FusionReranker, KeywordOverlapReranker
from retrieval import (
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


class RAGWorkflow:
    """End-to-end RAG workflow orchestrated by LangGraph."""

    def __init__(
        self,
        config: RAGConfig,
        *,
        llm: Optional[ChatGoogleGenerativeAI] = None,
        embeddings: Optional[GoogleGenerativeAIEmbeddings] = None,
        vector_store: Optional[Chroma] = None,
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

        self.vector_store = vector_store
        self.pinecone_index_manager = pinecone_manager
        self.pinecone_pipeline = pinecone_pipeline
        self.pinecone_retriever = pinecone_retriever
        self.sentence_window_retriever = sentence_window_retriever
        self.graph_retriever = graph_retriever
        self._retriever_dispatch: Dict[str, Callable[[str, float], List[Document]]] = {}
        self._retriever_order: List[str] = []
        self._event_handlers: List[Callable[[str, Dict[str, Any]], None]] = []

        if self.vector_store is None:
            self._initialize_vector_store()

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
        try:
            self.vector_store = Chroma(
                persist_directory=self.config.vector_store_path,
                embedding_function=self.embeddings,
            )
        except Exception as error:
            self.logger.warning("Error initializing vector store: %s", error)
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.config.vector_store_path,
            )

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

        if self.config.enable_pinecone_retriever:
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
                self._register_retriever("pinecone", self._pinecone_retrieval)
            except Exception as error:
                self.logger.warning("Failed to initialize Pinecone retriever: %s", error)
                self.pinecone_index_manager = None
                self.pinecone_pipeline = None
                self.pinecone_retriever = None

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
        if not self.vector_store:
            return 0

        collection = getattr(self.vector_store, "_collection", None)
        if collection and hasattr(collection, "count"):
            try:
                return int(collection.count())
            except Exception:
                return 0
        return 0

    def _ensure_vector_store_seeded(self) -> None:
        if not self.vector_store:
            return

        if self._get_index_count() == 0:
            self._load_sample_documents()

    def intake_query(self, state: RAGState) -> RAGState:
        question = state.get("question", "").strip()
        normalized_question = " ".join(question.split())

        status_messages = list(state.get("status_messages", []))
        status_messages.append("📥 Received query for processing")

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
            status_messages.append("📂 Knowledge index empty - loading sample corpus")
            self._load_sample_documents()
            current_count = self._get_index_count()
        else:
            status_messages.append(f"📦 Knowledge index ready with {current_count} documents")

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
            f"🔀 Fan-out to retrievers: {', '.join(active_retrievers)}"
        )

        retriever_results = dict(state.get("retriever_results", {}))
        errors = list(state.get("errors", []))

        for retriever_name in active_retrievers:
            try:
                documents = self._run_retriever(retriever_name, state, retriever_weights)
                retriever_results[retriever_name] = documents
                status_messages.append(
                    f"🔎 {retriever_name.replace('_', ' ').title()} retriever returned {len(documents)} documents"
                )
                self._emit_event(
                    "retriever.success",
                    {"name": retriever_name, "count": len(documents)},
                )
            except Exception as error:  # pragma: no cover - defensive branch
                message = f"❌ {retriever_name.replace('_', ' ').title()} retriever failed: {error}"
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
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")

        effective_weight = max(weight, 0.0)
        if effective_weight <= 0:
            return []

        top_k = max(1, min(int(self.config.retriever_top_k * max(effective_weight, 0.1)), 50))
        documents: List[Document] = []

        with_score = getattr(self.vector_store, "similarity_search_with_score", None)
        if callable(with_score):
            results = with_score(question, k=top_k)
            for doc, score in results:
                metadata = dict(doc.metadata or {})
                try:
                    metadata["score"] = float(score)
                except (TypeError, ValueError):
                    metadata["score"] = score
                metadata["retriever"] = "semantic"
                documents.append(Document(page_content=doc.page_content, metadata=metadata))
            return documents

        raw_docs = self.vector_store.similarity_search(question, k=top_k)
        for doc in raw_docs:
            metadata = dict(doc.metadata or {})
            metadata.setdefault("score", 0.0)
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
            f"🧮 Fused {fusion_result.total_candidates} documents into "
            f"{len(fused_documents)} ranked results"
        )
        if fusion_result.truncated:
            message += f" (trimmed {len(fusion_result.truncated)} for token budget)"
        status_messages.append(message)

        if fusion_result.metadata.get("reranker_applied"):
            status_messages.append("📊 Fusion reranker applied to refine ordering")

        token_budget = fusion_result.metadata.get("token_budget")
        if token_budget is not None:
            usage_message = f"⚖️ Context token usage {fusion_result.token_usage}"
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
        message = "⚠️ No relevant documents found. Returning fallback guidance."
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

    def generate_context(self, state: RAGState) -> RAGState:
        documents = state.get("fused_documents", [])
        status_messages = list(state.get("status_messages", []))
        status_messages.append("📝 Generating contextual summary from fused documents")

        context_parts: List[str] = []
        citations: List[Dict[str, Any]] = []

        for index, document in enumerate(documents, start=1):
            context_parts.append(f"[{index}] {document.page_content}")
            metadata = document.metadata or {}
            citations.append(
                {
                    "id": index,
                    "source": metadata.get("source", "Unknown"),
                    "page": metadata.get("page", "N/A"),
                    "content": (
                        document.page_content[:200] + "..."
                        if len(document.page_content) > 200
                        else document.page_content
                    ),
                }
            )

        context = "\n\n".join(context_parts)
        status_messages.append(f"✓ Generated context with {len(citations)} citations")

        return {
            **state,
            "context": context,
            "citations": citations,
            "status_messages": status_messages,
        }

    def generate_answer(self, state: RAGState) -> RAGState:
        status_messages = list(state.get("status_messages", []))
        status_messages.append("🤖 Generating answer with Gemini")

        prompt = (
            "You are a helpful enterprise Q&A assistant. Answer the question based on the provided context.\n"
            "Include citation numbers [1], [2], etc. in your answer to reference the sources.\n\n"
            f"Context:\n{state.get('context', '')}\n\n"
            f"Question: {state.get('question', '')}\n\n"
            "Answer (with citations):"
        )

        try:
            response = self.llm.invoke(prompt)
            answer = getattr(response, "content", str(response))
            status_messages.append("✓ Answer generated successfully")
            self._emit_event("llm.success", {"model": "gemini-pro"})
        except Exception as error:
            answer = f"Error generating answer: {error}"
            status_messages.append(f"❌ LLM generation failed: {error}")
            self._emit_event("llm.error", {"error": str(error)})

        return {**state, "answer": answer, "status_messages": status_messages}

    def format_response(self, state: RAGState) -> RAGState:
        status_messages = list(state.get("status_messages", []))
        answer = state.get("answer", "").strip()

        if not answer:
            answer = (
                "I was unable to produce an answer. Please try rephrasing the question or ingesting more data."
            )
            status_messages.append("ℹ️ Returned fallback answer due to empty response")
        else:
            status_messages.append("✅ Response ready for presentation")

        citations = state.get("citations", [])
        for index, citation in enumerate(citations, start=1):
            citation["id"] = index

        self._emit_event(
            "workflow.complete",
            {
                "citations": len(citations),
                "fallback": state.get("fallback_reason") is not None,
            },
        )

        return {
            **state,
            "answer": answer,
            "citations": citations,
            "status_messages": status_messages,
        }

    def _build_workflow(self):
        graph = StateGraph(RAGState)

        def register_node(name: str, handler: Callable[[RAGState], RAGState]) -> None:
            graph.add_node(name, handler)
            self._graph_nodes.append(name)

        def register_edge(source: str, target: str, condition: Optional[str] = None) -> None:
            graph.add_edge(source, target)
            self._graph_edges.append({"source": source, "target": target, "condition": condition})

        register_node("intake_query", self.intake_query)
        register_node("ensure_ingestion_ready", self.ensure_ingestion_ready)
        register_node("multi_retriever_fanout", self.multi_retriever_fanout)
        register_node("fuse_and_rank", self.fuse_and_rank)
        register_node("handle_no_results", self.handle_no_results)
        register_node("generate_context", self.generate_context)
        register_node("generate_answer", self.generate_answer)
        register_node("format_response", self.format_response)

        graph.set_entry_point("intake_query")

        register_edge("intake_query", "ensure_ingestion_ready")
        register_edge("ensure_ingestion_ready", "multi_retriever_fanout")
        register_edge("multi_retriever_fanout", "fuse_and_rank")

        graph.add_conditional_edges(
            "fuse_and_rank",
            self._route_post_fusion,
            {
                "continue": "generate_context",
                "fallback": "handle_no_results",
            },
        )
        self._graph_edges.append(
            {"source": "fuse_and_rank", "target": "generate_context", "condition": "continue"}
        )
        self._graph_edges.append(
            {"source": "fuse_and_rank", "target": "handle_no_results", "condition": "fallback"}
        )

        register_edge("handle_no_results", "format_response")
        register_edge("generate_context", "generate_answer")
        register_edge("generate_answer", "format_response")
        register_edge("format_response", END)

        return graph.compile()

    def describe_graph(self) -> Dict[str, Sequence[Dict[str, Optional[str]]]]:
        return {
            "nodes": [{"name": node} for node in self._graph_nodes],
            "edges": self._graph_edges,
        }

    def run(self, question: str, retriever_weights: Optional[Dict[str, float]] = None) -> RAGState:
        weights = dict(retriever_weights or {})
        initial_state: RAGState = {
            "question": question,
            "normalized_question": "",
            "documents": [],
            "fused_documents": [],
            "context": "",
            "answer": "",
            "citations": [],
            "status_messages": [],
            "retriever_weights": weights,
            "retriever_results": {},
            "fusion_diagnostics": {},
            "errors": [],
            "active_retrievers": [],
        }

        return self.workflow.invoke(initial_state)

    def stream(self, question: str, retriever_weights: Optional[Dict[str, float]] = None):
        weights = dict(retriever_weights or {})
        initial_state: RAGState = {
            "question": question,
            "normalized_question": "",
            "documents": [],
            "fused_documents": [],
            "context": "",
            "answer": "",
            "citations": [],
            "status_messages": [],
            "retriever_weights": weights,
            "retriever_results": {},
            "fusion_diagnostics": {},
            "errors": [],
            "active_retrievers": [],
        }

        for state in self.workflow.stream(initial_state):
            yield state

    def ingest_documents(self, documents: List[Document]) -> int:
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")

        if not documents:
            return 0

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        original_documents = list(documents)
        split_docs = text_splitter.split_documents(original_documents)

        if split_docs:
            self.vector_store.add_documents(split_docs)
            if hasattr(self.vector_store, "persist"):
                self.vector_store.persist()

        if self.pinecone_pipeline and split_docs:
            try:
                if self.pinecone_index_manager:
                    self.pinecone_index_manager.ensure_index(metric=self.config.pinecone_metric)
                self.pinecone_pipeline.upsert_documents(split_docs)
            except Exception as error:  # pragma: no cover - defensive
                self.logger.warning("Pinecone upsert failed: %s", error)

        if self.sentence_window_retriever:
            try:
                self.sentence_window_retriever.index_documents(original_documents)
            except Exception as error:  # pragma: no cover - defensive
                self.logger.warning("Sentence window indexing failed: %s", error)

        if self.graph_retriever:
            try:
                self.graph_retriever.index_documents(original_documents)
            except Exception as error:  # pragma: no cover - defensive
                self.logger.warning("Graph indexing failed: %s", error)

        self._emit_event(
            "ingestion.custom",
            {"documents": len(original_documents), "chunks": len(split_docs)},
        )
        return len(split_docs)

    def refresh_index(self) -> bool:
        if self.vector_store and hasattr(self.vector_store, "persist"):
            self.vector_store.persist()
            self._emit_event("ingestion.refresh", {"success": True})
            return True
        self._emit_event("ingestion.refresh", {"success": False})
        return False
