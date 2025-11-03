import os
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from operator import add
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from config import RAGConfig


class RAGState(TypedDict):
    question: str
    documents: List[Document]
    context: str
    answer: str
    citations: List[Dict[str, Any]]
    status_messages: Annotated[Sequence[str], add]
    retriever_weights: Dict[str, float]


class RAGWorkflow:
    def __init__(self, config: RAGConfig):
        self.config = config
        os.environ["GOOGLE_API_KEY"] = config.google_api_key
        
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        
        self.vector_store = None
        self._initialize_vector_store()
        
        self.workflow = self._build_workflow()
    
    def _initialize_vector_store(self):
        try:
            self.vector_store = Chroma(
                persist_directory=self.config.vector_store_path,
                embedding_function=self.embeddings
            )
            
            if self.vector_store._collection.count() == 0:
                self._load_sample_documents()
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.config.vector_store_path
            )
            self._load_sample_documents()
    
    def _load_sample_documents(self):
        sample_docs = [
            Document(
                page_content="Enterprise software architecture involves designing scalable, maintainable, and secure systems. "
                "Key principles include separation of concerns, modularity, and adherence to design patterns.",
                metadata={"source": "architecture_guide.pdf", "page": 1}
            ),
            Document(
                page_content="Microservices architecture breaks down applications into smaller, independent services. "
                "Each service handles a specific business capability and communicates via APIs.",
                metadata={"source": "microservices_handbook.pdf", "page": 3}
            ),
            Document(
                page_content="Cloud computing provides on-demand access to computing resources. "
                "Major cloud providers include AWS, Azure, and Google Cloud Platform. "
                "Benefits include scalability, cost-efficiency, and global reach.",
                metadata={"source": "cloud_basics.pdf", "page": 5}
            ),
            Document(
                page_content="DevOps practices combine development and operations to improve deployment frequency. "
                "Key practices include continuous integration, continuous delivery, and infrastructure as code.",
                metadata={"source": "devops_guide.pdf", "page": 2}
            ),
            Document(
                page_content="Security best practices for enterprise applications include authentication, authorization, "
                "encryption, input validation, and regular security audits. Zero-trust architecture is becoming standard.",
                metadata={"source": "security_handbook.pdf", "page": 7}
            ),
        ]
        
        if self.vector_store:
            self.vector_store.add_documents(sample_docs)
            if hasattr(self.vector_store, 'persist'):
                self.vector_store.persist()
    
    def retrieve_documents(self, state: RAGState) -> RAGState:
        question = state["question"]
        retriever_weights = state.get("retriever_weights", {"semantic": 1.0})
        
        status_messages = list(state.get("status_messages", []))
        status_messages.append(f"🔍 Retrieving documents with weights: {retriever_weights}")
        
        top_k = int(self.config.retriever_top_k * retriever_weights.get("semantic", 1.0))
        top_k = max(1, min(top_k, 20))
        
        documents = self.vector_store.similarity_search(question, k=top_k)
        
        status_messages.append(f"✓ Retrieved {len(documents)} relevant documents")
        
        return {
            **state,
            "documents": documents,
            "status_messages": status_messages
        }
    
    def generate_context(self, state: RAGState) -> RAGState:
        documents = state["documents"]
        
        status_messages = list(state.get("status_messages", []))
        status_messages.append("📝 Generating context from retrieved documents")
        
        context_parts = []
        citations = []
        
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[{i}] {doc.page_content}")
            citations.append({
                "id": i,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        
        context = "\n\n".join(context_parts)
        
        status_messages.append(f"✓ Generated context with {len(citations)} citations")
        
        return {
            **state,
            "context": context,
            "citations": citations,
            "status_messages": status_messages
        }
    
    def generate_answer(self, state: RAGState) -> RAGState:
        question = state["question"]
        context = state["context"]
        
        status_messages = list(state.get("status_messages", []))
        status_messages.append("🤖 Generating answer with Gemini")
        
        prompt = f"""You are a helpful enterprise Q&A assistant. Answer the question based on the provided context.
Include citation numbers [1], [2], etc. in your answer to reference the sources.

Context:
{context}

Question: {question}

Answer (with citations):"""
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content
            status_messages.append("✓ Answer generated successfully")
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
            status_messages.append(f"❌ Error: {str(e)}")
        
        return {
            **state,
            "answer": answer,
            "status_messages": status_messages
        }
    
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(RAGState)
        
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("context", self.generate_context)
        workflow.add_node("answer", self.generate_answer)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "context")
        workflow.add_edge("context", "answer")
        workflow.add_edge("answer", END)
        
        return workflow.compile()
    
    def run(self, question: str, retriever_weights: Dict[str, float] = None) -> RAGState:
        if retriever_weights is None:
            retriever_weights = {"semantic": 1.0}
        
        initial_state: RAGState = {
            "question": question,
            "documents": [],
            "context": "",
            "answer": "",
            "citations": [],
            "status_messages": [],
            "retriever_weights": retriever_weights
        }
        
        result = self.workflow.invoke(initial_state)
        return result
    
    def stream(self, question: str, retriever_weights: Dict[str, float] = None):
        if retriever_weights is None:
            retriever_weights = {"semantic": 1.0}
        
        initial_state: RAGState = {
            "question": question,
            "documents": [],
            "context": "",
            "answer": "",
            "citations": [],
            "status_messages": [],
            "retriever_weights": retriever_weights
        }
        
        for state in self.workflow.stream(initial_state):
            yield state
    
    def ingest_documents(self, documents: List[Document]) -> int:
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        split_docs = text_splitter.split_documents(documents)
        self.vector_store.add_documents(split_docs)
        
        if hasattr(self.vector_store, 'persist'):
            self.vector_store.persist()
        
        return len(split_docs)
    
    def refresh_index(self):
        if self.vector_store and hasattr(self.vector_store, 'persist'):
            self.vector_store.persist()
            return True
        return False
