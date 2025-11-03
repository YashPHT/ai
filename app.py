import streamlit as st
from typing import Dict
from langchain.schema import Document
from config import RAGConfig
from rag_workflow import RAGWorkflow

st.set_page_config(
    page_title="Enterprise RAG Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def authentication_hook():
    if st.session_state.config.enable_auth:
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        
        if not st.session_state.authenticated:
            st.title("🔐 Authentication Required")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                if username == "admin" and password == "admin":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            
            st.stop()
    return True


def initialize_session_state():
    if "config" not in st.session_state:
        st.session_state.config = RAGConfig.from_env()
    
    if "workflow" not in st.session_state:
        try:
            st.session_state.workflow = RAGWorkflow(st.session_state.config)
        except Exception as e:
            st.error(f"Failed to initialize RAG workflow: {e}")
            st.stop()
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    if "retriever_weights" not in st.session_state:
        st.session_state.retriever_weights = {"semantic": 1.0}


def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        st.header("Environment")
        st.info(f"Mode: {st.session_state.config.environment}")
        
        st.header("Retriever Settings")
        
        semantic_weight = st.slider(
            "Semantic Search Weight",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.retriever_weights.get("semantic", 1.0),
            step=0.1,
            help="Adjust the weight for semantic retrieval"
        )
        
        st.session_state.retriever_weights = {"semantic": semantic_weight}
        
        top_k = st.number_input(
            "Top K Documents",
            min_value=1,
            max_value=20,
            value=st.session_state.config.retriever_top_k,
            help="Number of documents to retrieve"
        )
        
        st.session_state.config.retriever_top_k = top_k
        
        st.divider()
        
        st.header("🛠️ Admin Controls")
        
        if st.button("🔄 Refresh Index", use_container_width=True):
            with st.spinner("Refreshing index..."):
                success = st.session_state.workflow.refresh_index()
                if success:
                    st.success("Index refreshed successfully!")
                else:
                    st.warning("Index refresh not available")
        
        st.subheader("Document Ingestion")
        
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=["txt"],
            help="Upload text files to add to the knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("📥 Ingest Document", use_container_width=True):
                with st.spinner("Ingesting document..."):
                    try:
                        content = uploaded_file.read().decode("utf-8")
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": uploaded_file.name,
                                "type": "user_upload"
                            }
                        )
                        
                        num_chunks = st.session_state.workflow.ingest_documents([doc])
                        st.success(f"✅ Ingested {num_chunks} document chunks!")
                    except Exception as e:
                        st.error(f"Error ingesting document: {e}")
        
        st.divider()
        
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()


def render_main_interface():
    st.title("🤖 Enterprise RAG Q&A System")
    st.markdown("Ask questions and get answers with citations from the knowledge base.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask a Question")
        
        question = st.text_area(
            "Your Question",
            placeholder="e.g., What are the key principles of microservices architecture?",
            height=100,
            label_visibility="collapsed"
        )
        
        col_submit, col_example = st.columns([1, 2])
        
        with col_submit:
            submit_button = st.button("🚀 Submit", type="primary", use_container_width=True)
        
        with col_example:
            if st.button("💡 Try Example", use_container_width=True):
                question = "What are the key security practices for enterprise applications?"
                submit_button = True
        
        if submit_button and question:
            with st.spinner("Processing your question..."):
                status_container = st.container()
                
                with status_container:
                    st.subheader("📊 Processing Status")
                    status_placeholder = st.empty()
                
                try:
                    result = st.session_state.workflow.invoke(
                        question
                    )
                    
                    status_messages = result.get("status_messages", [])
                    with status_placeholder:
                        for msg in status_messages:
                            st.text(msg)
                    
                    st.session_state.conversation_history.insert(0, {
                        "question": question,
                        "answer": result.get("answer", ""),
                        "citations": result.get("citations", []),
                        "context": result.get("context", "")
                    })
                    
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Error processing question: {e}")
    
    with col2:
        st.header("📚 System Info")
        
        config_expander = st.expander("Configuration Details", expanded=False)
        with config_expander:
            st.json({
                "Environment": st.session_state.config.environment,
                "Top K": st.session_state.config.retriever_top_k,
                "Chunk Size": st.session_state.config.chunk_size,
                "Chunk Overlap": st.session_state.config.chunk_overlap,
                "Retriever Weights": st.session_state.retriever_weights
            })
        
        st.info("💡 Adjust retriever weights and settings in the sidebar")
    
    st.divider()
    
    if st.session_state.conversation_history:
        st.header("📜 Recent Queries")
        
        for idx, entry in enumerate(st.session_state.conversation_history):
            with st.expander(f"Q: {entry['question'][:80]}...", expanded=(idx == 0)):
                st.subheader("❓ Question")
                st.write(entry["question"])
                
                st.subheader("✅ Answer")
                st.write(entry["answer"])
                
                st.subheader("📎 Citations & Sources")
                
                if entry.get("citations"):
                    for citation in entry["citations"]:
                        with st.container():
                            st.markdown(f"""
                            **Source:** `{citation.get('source', 'N/A')}` (Page {citation.get('page', 'N/A')})
                            """)
                else:
                    st.info("No citations available")
                
                context_expander = st.expander("🔍 View Full Context", expanded=False)
                with context_expander:
                    st.text_area(
                        "Context",
                        value=entry.get("context", "No context available"),
                        height=200,
                        label_visibility="collapsed",
                        disabled=True
                    )
    else:
        st.info("👆 Ask a question above to get started!")


def main():
    try:
        initialize_session_state()
        
        is_valid, error_msg = st.session_state.config.validate()
        if not is_valid:
            st.error(f"❌ Configuration Error: {error_msg}")
            st.info("Please set up your environment variables. See `.env.example` for reference.")
            st.stop()
        
        authentication_hook()
        
        render_sidebar()
        render_main_interface()
        
    except Exception as e:
        st.error(f"Application Error: {e}")
        st.info("Please check your configuration and try again.")


if __name__ == "__main__":
    main()
