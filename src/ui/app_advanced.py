"""
Advanced Streamlit Web UI for Production RAG System

Features:
- Multiple collections/folders for organizing documents
- View PDFs directly in browser
- Collection-specific querying
- Document management (list, view, delete)
- Interactive chat interface
- Source citations and quality metrics
"""

import streamlit as st
import tempfile
import shutil
import base64
from pathlib import Path
from datetime import datetime
import time
import json

import sys
from pathlib import Path as PathLib
sys.path.insert(0, str(PathLib(__file__).parent.parent.parent))

from main import RAGSystem
from src.ingestion.document_processor import DocumentProcessor
from src.rag.config import get_default_config
import chromadb

# Page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .collection-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        color: #000000;
    }
    .document-item {
        background-color: #ffffff;
        padding: 0.75rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        cursor: pointer;
        color: #000000;
    }
    .document-item:hover {
        background-color: #f5f5f5;
        border-color: #1f77b4;
        color: #000000;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #000000;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
        color: #000000;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 3px solid #ff9800;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #000000;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        color: #000000;
    }
    /* Fix cursor for selectbox and interactive elements */
    .stSelectbox > div > div {
        cursor: default !important;
    }
    .stSelectbox label {
        cursor: default !important;
    }
</style>
""", unsafe_allow_html=True)


def get_collection_list():
    """Get list of all collections from data/documents/ folder."""
    try:
        documents_dir = Path("./data/documents")
        if not documents_dir.exists():
            documents_dir.mkdir(parents=True, exist_ok=True)
            return []
        
        # List all subdirectories in data/documents/
        collections = [d.name for d in documents_dir.iterdir() if d.is_dir()]
        return sorted(collections)
    except:
        return []


def get_collection_metadata(collection_name):
    """Get metadata for a collection from file system."""
    try:
        # Check file system for actual PDFs
        collection_folder = Path(f"./data/documents/{collection_name}")
        
        if not collection_folder.exists():
            return {'chunks': 0, 'documents': 0, 'files': [], 'indexed': 0}
        
        # Get PDFs from file system
        pdf_files = list(collection_folder.glob("*.pdf"))
        file_names = sorted([f.name for f in pdf_files])
        
        # Also check ChromaDB for indexed chunks
        indexed_chunks = 0
        try:
            persist_dir = "./data/vector_store"
            client = chromadb.PersistentClient(path=persist_dir)
            collection = client.get_collection(collection_name)
            indexed_chunks = collection.count()
        except:
            indexed_chunks = 0
        
        return {
            'chunks': indexed_chunks,
            'documents': len(pdf_files),
            'files': file_names,
            'indexed': indexed_chunks
        }
    except Exception as e:
        return {'chunks': 0, 'documents': 0, 'files': [], 'indexed': 0}


def create_new_collection(collection_name):
    """Create a new collection with physical folder in data/documents/."""
    try:
        # Create physical folder for the collection in data/documents/
        collection_folder = Path(f"./data/documents/{collection_name}")
        collection_folder.mkdir(parents=True, exist_ok=True)
        
        # Create ChromaDB collection
        persist_dir = "./data/vector_store"
        client = chromadb.PersistentClient(path=persist_dir)
        
        # Try to get existing collection or create new one
        try:
            client.get_collection(name=collection_name)
        except:
            client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        return True
    except Exception as e:
        st.error(f"Error creating collection: {e}")
        return False


def rename_collection(old_name, new_name):
    """
    Rename a collection (both ChromaDB and physical folder).
    
    Args:
        old_name: Current collection name
        new_name: New collection name
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import shutil
        
        # Sanitize new name
        new_name = new_name.strip().lower().replace(" ", "_")
        
        if not new_name:
            st.error("New name cannot be empty")
            return False
        
        if new_name == old_name:
            st.warning("New name is the same as old name")
            return False
        
        # Check if new name already exists
        new_folder = Path(f"./data/documents/{new_name}")
        if new_folder.exists():
            st.error(f"Collection '{new_name}' already exists")
            return False
        
        # Rename physical folder
        old_folder = Path(f"./data/documents/{old_name}")
        if old_folder.exists():
            old_folder.rename(new_folder)
        
        # Rename ChromaDB collection (copy data to new collection, delete old)
        persist_dir = "./data/vector_store"
        client = chromadb.PersistentClient(path=persist_dir)
        
        try:
            # Get old collection
            old_collection = client.get_collection(name=old_name)
            
            # Get all data from old collection
            results = old_collection.get(include=['embeddings', 'documents', 'metadatas'])
            
            # Create new collection
            new_collection = client.create_collection(
                name=new_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Copy data to new collection if there is any
            if results['ids']:
                new_collection.add(
                    ids=results['ids'],
                    embeddings=results['embeddings'],
                    documents=results['documents'],
                    metadatas=results['metadatas']
                )
            
            # Delete old collection
            client.delete_collection(name=old_name)
            
            # Clear Streamlit cache for RAG system
            get_rag_system.clear()
            
        except Exception as e:
            # If ChromaDB operations fail, revert folder rename
            if new_folder.exists() and not old_folder.exists():
                new_folder.rename(old_folder)
            raise e
        
        return True
        
    except Exception as e:
        st.error(f"Error renaming collection: {e}")
        return False


def delete_collection(collection_name):
    """
    Delete a collection (both ChromaDB and physical folder).
    
    Args:
        collection_name: Name of collection to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import shutil
        
        # Delete physical folder
        collection_folder = Path(f"./data/documents/{collection_name}")
        if collection_folder.exists():
            shutil.rmtree(collection_folder)
        
        # Delete ChromaDB collection
        persist_dir = "./data/vector_store"
        client = chromadb.PersistentClient(path=persist_dir)
        
        try:
            client.delete_collection(name=collection_name)
        except:
            pass  # Collection might not exist in DB
        
        # Clear Streamlit cache for RAG system
        get_rag_system.clear()
        
        return True
        
    except Exception as e:
        st.error(f"Error deleting collection: {e}")
        return False


def index_all_documents(collection_name, clear_existing=False):
    """
    Index all PDF documents in a collection folder.
    
    Args:
        collection_name: Name of collection to index
        clear_existing: If True, clear existing chunks before re-indexing
        
    Returns:
        tuple: (success, total_chunks_processed)
    """
    try:
        collection_folder = Path(f"./data/documents/{collection_name}")
        
        if not collection_folder.exists():
            return (False, 0)
        
        # Get all PDF files
        pdf_files = list(collection_folder.glob("*.pdf"))
        
        if not pdf_files:
            return (False, 0)
        
        # Clear existing chunks if requested
        if clear_existing:
            try:
                persist_dir = "./data/vector_store"
                client = chromadb.PersistentClient(path=persist_dir)
                
                # Delete and recreate collection to clear all data
                try:
                    client.delete_collection(name=collection_name)
                except:
                    pass  # Collection might not exist
                
                # Recreate collection with correct settings
                client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                
                # Clear Streamlit cache for RAG system
                get_rag_system.clear()
                    
            except Exception as e:
                st.error(f"Error clearing collection: {e}")
                return (False, 0)
        
        # Get RAG system for this collection (will use new empty collection if cleared)
        rag_system = get_rag_system(collection_name)
        
        # Process all documents
        total_chunks = rag_system.ingest_documents_from_directory(collection_folder)
        
        return (True, total_chunks)
        
    except Exception as e:
        st.error(f"Error indexing documents: {e}")
        return (False, 0)


def display_pdf(file_path):
    """Display PDF in browser using iframe."""
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        pdf_display = f'''
        <iframe src="data:application/pdf;base64,{base64_pdf}" 
                width="100%" height="800" type="application/pdf">
        </iframe>
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")


def display_message(role, content, metadata=None):
    """Display a chat message."""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🙋 You:</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 Assistant:</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)
        
        if metadata:
            with st.expander("📊 View Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", f"{metadata['confidence']:.1%}")
                with col2:
                    st.metric("Retrieved Docs", metadata['retrieved_docs'])
                with col3:
                    st.metric("Query Time", f"{metadata['query_time']:.2f}s")
                
                if metadata.get('sources'):
                    st.markdown("**📚 Sources:**")
                    for i, src in enumerate(metadata['sources'][:3], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            [{i}] <strong>{src['source']}</strong> 
                            (Page {src['page']}, Relevance: {src['score']:.1%})
                        </div>
                        """, unsafe_allow_html=True)
                
                if metadata.get('evaluation'):
                    eval_data = metadata['evaluation']
                    st.markdown("**🎯 Quality Scores:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Faithfulness", f"{eval_data.get('faithfulness', 0):.1%}")
                    with col2:
                        st.metric("Relevance", f"{eval_data.get('relevance', 0):.1%}")
                    with col3:
                        st.metric("Overall", f"{eval_data.get('overall_score', 0):.1%}")


@st.cache_resource
def get_rag_system(collection_name: str):
    """
    Get RAG system instance for specific collection.
    
    Args:
        collection_name: Name of the collection to use
        
    Returns:
        RAGSystem instance configured for the collection
    """
    return RAGSystem(collection_name=collection_name)


def main():
    """Main application."""
    
    # Header
    st.markdown('<h1 class="main-header">📚 RAG Document Assistant</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session state
    if 'current_collection' not in st.session_state:
        # Get available collections
        collections = get_collection_list()
        if collections:
            st.session_state.current_collection = collections[0]  # Use first available
        else:
            # Create default collection if none exist
            default_collection = "my_documents"
            create_new_collection(default_collection)
            st.session_state.current_collection = default_collection
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}
    
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = "chat"  # chat, documents, viewer
    
    if 'selected_pdf' not in st.session_state:
        st.session_state.selected_pdf = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Collections")
        
        # Get existing collections
        collections = get_collection_list()
        if not collections:
            collections = []
        
        # Collection selector
        if collections:
            # Make sure current collection is in the list
            if st.session_state.current_collection not in collections:
                st.session_state.current_collection = collections[0]
            
            selected_collection = st.selectbox(
                "Select Collection",
                collections,
                index=collections.index(st.session_state.current_collection)
            )
            st.session_state.current_collection = selected_collection
        else:
            st.info("No collections found. Create one below!")
            # Create default if none exist
            if st.button("Create Default Collection"):
                create_new_collection("my_documents")
                st.rerun()
        
        # Create new collection
        with st.expander("➕ Create New Collection"):
            new_collection_name = st.text_input(
                "Collection Name",
                placeholder="e.g., invoices, contracts, reports",
                key="new_collection_input"
            )
            if st.button("Create Collection", type="primary", key="create_btn"):
                if new_collection_name:
                    # Sanitize collection name
                    clean_name = new_collection_name.strip().lower().replace(" ", "_")
                    if create_new_collection(clean_name):
                        st.success(f"✅ Created collection: {clean_name}")
                        st.session_state.current_collection = clean_name
                        st.rerun()
                else:
                    st.error("Please enter a collection name")
        
        # Rename collection
        if collections and st.session_state.current_collection:
            with st.expander("✏️ Rename Collection"):
                st.markdown(f"**Current:** `{st.session_state.current_collection}`")
                renamed_collection = st.text_input(
                    "New Name",
                    placeholder="e.g., my_invoices",
                    key="rename_collection_input"
                )
                if st.button("Rename", type="secondary", key="rename_btn"):
                    if renamed_collection:
                        old_name = st.session_state.current_collection
                        if rename_collection(old_name, renamed_collection):
                            st.success(f"✅ Renamed '{old_name}' to '{renamed_collection}'")
                            st.session_state.current_collection = renamed_collection
                            # Clear chat history for old collection name
                            if old_name in st.session_state.chat_history:
                                st.session_state.chat_history[renamed_collection] = st.session_state.chat_history.pop(old_name)
                            st.rerun()
                    else:
                        st.error("Please enter a new name")
        
        # Delete collection (GitHub-style confirmation)
        if collections and st.session_state.current_collection:
            with st.expander("🗑️ Delete Collection"):
                st.warning("⚠️ **Warning:** This action cannot be undone!")
                st.markdown(f"**Collection to delete:** `{st.session_state.current_collection}`")
                
                # Initialize delete confirmation state
                if 'delete_confirmation' not in st.session_state:
                    st.session_state.delete_confirmation = ""
                
                st.markdown(f"Type **`{st.session_state.current_collection}`** to confirm:")
                
                confirmation_input = st.text_input(
                    "Confirmation",
                    placeholder=f"Type {st.session_state.current_collection}",
                    key="delete_confirmation_input",
                    label_visibility="collapsed"
                )
                
                # Check if confirmation matches
                confirmation_matches = confirmation_input == st.session_state.current_collection
                
                if st.button(
                    "🗑️ Delete Collection", 
                    type="secondary",
                    disabled=not confirmation_matches,
                    key="delete_btn"
                ):
                    if confirmation_matches:
                        collection_to_delete = st.session_state.current_collection
                        
                        # Delete the collection
                        if delete_collection(collection_to_delete):
                            st.success(f"✅ Deleted collection: {collection_to_delete}")
                            
                            # Clear chat history for deleted collection
                            if collection_to_delete in st.session_state.chat_history:
                                del st.session_state.chat_history[collection_to_delete]
                            
                            # Switch to another collection or create default
                            remaining_collections = get_collection_list()
                            if remaining_collections:
                                st.session_state.current_collection = remaining_collections[0]
                            else:
                                # No collections left, create default
                                create_new_collection("my_documents")
                                st.session_state.current_collection = "my_documents"
                            
                            st.rerun()
                    else:
                        st.error("Collection name doesn't match!")
                
                if not confirmation_matches and confirmation_input:
                    st.error("❌ Name doesn't match. Please type the exact collection name.")
        
        st.markdown("---")
        
        # Current collection info
        if st.session_state.current_collection:
            st.markdown(f"### 📊 Current: `{st.session_state.current_collection}`")
            metadata = get_collection_metadata(st.session_state.current_collection)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", metadata['documents'])
            with col2:
                st.metric("Chunks", metadata['chunks'])
            
            # Show indexing/re-indexing button when documents exist
            if metadata['documents'] > 0:
                # Check if already indexed
                already_indexed = metadata['indexed'] > 0
                
                if already_indexed:
                    # Re-index button (will clear and re-index)
                    button_text = "🔄 Re-index All Documents"
                    button_help = f"Clear existing {metadata['chunks']} chunks and re-index all {metadata['documents']} documents"
                    spinner_text = f"Clearing and re-indexing {metadata['documents']} documents..."
                else:
                    # Initial index button
                    st.warning(f"⚠️ {metadata['documents']} PDF(s) found but not indexed yet!")
                    button_text = "🔄 Index All Documents"
                    button_help = f"Index {metadata['documents']} documents"
                    spinner_text = f"Indexing {metadata['documents']} documents..."
                
                # Show the button
                if st.button(
                    button_text, 
                    type="primary" if not already_indexed else "secondary",
                    use_container_width=True, 
                    key="index_all_btn",
                    help=button_help
                ):
                    with st.spinner(spinner_text):
                        # Clear existing if already indexed
                        success, total_chunks = index_all_documents(
                            st.session_state.current_collection,
                            clear_existing=already_indexed
                        )
                        
                        if success:
                            action = "Re-indexed" if already_indexed else "Indexed"
                            st.success(f"✅ {action} {metadata['documents']} document(s) → {total_chunks} chunks!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Failed to index documents")
        
        st.markdown("---")
        
        # Upload documents
        st.markdown("### 📤 Upload Documents")
        
        # Initialize upload counter in session state (used to clear file uploader)
        if 'upload_counter' not in st.session_state:
            st.session_state.upload_counter = 0
        
        # File uploader with unique key that changes after upload
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDFs to current collection",
            key=f"file_uploader_{st.session_state.upload_counter}"
        )
        
        if uploaded_files:
            if st.button("Process Files", type="primary", key="process_files_btn"):
                with st.spinner("Processing documents..."):
                    # Get current collection name
                    current_collection = st.session_state.current_collection
                    
                    # Create collection-specific folder in data/documents/
                    collection_folder = Path(f"./data/documents/{current_collection}")
                    collection_folder.mkdir(parents=True, exist_ok=True)
                    
                    # Create temp directory
                    temp_dir = Path(tempfile.mkdtemp())
                    
                    try:
                        # Save files to temp and collection folder
                        for uploaded_file in uploaded_files:
                            # Save to temp for processing
                            temp_file_path = temp_dir / uploaded_file.name
                            with open(temp_file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Save to collection-specific folder in data/documents/
                            collection_file_path = collection_folder / uploaded_file.name
                            with open(collection_file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                        
                        # Process documents with collection-specific RAG system
                        rag_system = get_rag_system(current_collection)
                        count = rag_system.ingest_documents_from_directory(temp_dir)
                        
                        st.success(f"✅ Processed {count} chunks and saved to data/documents/{current_collection}/ folder!")
                        
                        # Increment upload counter to clear the file uploader on next rerun
                        st.session_state.upload_counter += 1
                        
                        st.rerun()
                    
                    finally:
                        shutil.rmtree(temp_dir, ignore_errors=True)
        
        st.markdown("---")
        
        # View mode selector
        st.markdown("### 🎯 View Mode")
        view_mode = st.radio(
            "Select View",
            ["💬 Chat", "📄 Documents", "📊 Statistics"],
            label_visibility="collapsed"
        )
        
        if "Chat" in view_mode:
            st.session_state.view_mode = "chat"
        elif "Documents" in view_mode:
            st.session_state.view_mode = "documents"
        else:
            st.session_state.view_mode = "stats"
    
    # Main content area
    if st.session_state.view_mode == "chat":
        show_chat_view()
    elif st.session_state.view_mode == "documents":
        show_documents_view()
    elif st.session_state.view_mode == "stats":
        show_statistics_view()


def show_chat_view():
    """Show chat interface."""
    st.markdown("### 💬 Chat with Your Documents")
    
    # Get or create chat history for current collection
    collection = st.session_state.current_collection
    if collection not in st.session_state.chat_history:
        st.session_state.chat_history[collection] = []
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history[collection]:
            display_message(
                message['role'],
                message['content'],
                message.get('metadata')
            )
    
    # Check if collection has documents
    metadata = get_collection_metadata(collection)
    
    # Show warning if files exist but not indexed
    if metadata['documents'] > 0 and metadata['indexed'] == 0:
        st.warning(f"⚠️ Found {metadata['documents']} PDF(s) in `{collection}` collection but they're not indexed yet!")
        st.info("👈 Use the '🔄 Index All Documents' button in the sidebar to process them.")
        
        # List the files
        with st.expander("📄 Files found (click to expand)"):
            for file in metadata['files']:
                st.markdown(f"  - {file}")
    
    # Show info if no files at all
    if metadata['documents'] == 0:
        st.info(f"👆 Upload documents to the `{collection}` collection to start chatting!")
        st.markdown("""
        **Quick Start:**
        1. Upload PDFs using the sidebar
        2. Click "Process Files"
        3. Start asking questions!
        """)
    
    # Always show chat interface if there are documents (indexed or not)
    if metadata['documents'] > 0:
        
        # Chat input with form for Enter key support
        with st.form(key=f"chat_form_{collection}", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                # Show different placeholder based on indexing status
                placeholder = "E.g., What is the total amount? (Press Enter to send)" if metadata['indexed'] > 0 else "⚠️ Please index the files first to query"
                user_question = st.text_input(
                    "Ask a question:",
                    key=f"user_input_{collection}",
                    placeholder=placeholder,
                    label_visibility="collapsed",
                    disabled=(metadata['indexed'] == 0)  # Disable if not indexed
                )
            
            with col2:
                ask_button = st.form_submit_button(
                    "Send 📤", 
                    type="primary", 
                    use_container_width=True,
                    disabled=(metadata['indexed'] == 0)  # Disable if not indexed
                )
        
        # Process question (triggered by Enter or button click)
        if ask_button and user_question and metadata['indexed'] > 0:
            # Add user message
            st.session_state.chat_history[collection].append({
                'role': 'user',
                'content': user_question
            })
            
            # Get response using collection-specific RAG system
            with st.spinner("🔍 Searching documents..."):
                try:
                    rag_system = get_rag_system(collection)
                    response = rag_system.query(user_question, evaluate=True)
                    
                    # Add assistant message
                    st.session_state.chat_history[collection].append({
                        'role': 'assistant',
                        'content': response['answer'],
                        'metadata': {
                            'confidence': response['confidence'],
                            'retrieved_docs': response['retrieved_docs'],
                            'query_time': response['query_time'],
                            'sources': response.get('sources', []),
                            'evaluation': response.get('evaluation')
                        }
                    })
                    
                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


def show_documents_view():
    """Show documents list and viewer."""
    st.markdown("### 📄 Documents")
    
    collection = st.session_state.current_collection
    metadata = get_collection_metadata(collection)
    
    if metadata['documents'] == 0:
        st.info(f"No documents in `{collection}` collection yet.")
        return
    
    st.markdown(f"**Collection:** `{collection}` | **Documents:** {metadata['documents']} | **Chunks:** {metadata['chunks']}")
    st.markdown(f"**Folder:** `data/documents/{collection}/`")
    
    # Show warning banner if not indexed
    if metadata['documents'] > 0 and metadata['indexed'] == 0:
        st.warning(f"⚠️ {metadata['documents']} PDF(s) found but not indexed yet. Use 'Index All Documents' button in the sidebar to process them.")
    
    st.markdown("---")
    
    # Create two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Document List")
        
        # List all documents
        for i, doc_name in enumerate(metadata['files']):
            # Create row with view button and delete button
            doc_col1, doc_col2 = st.columns([4, 1])
            
            with doc_col1:
                if st.button(f"📄 {doc_name}", key=f"doc_{i}", use_container_width=True):
                    st.session_state.selected_pdf = doc_name
            
            with doc_col2:
                if st.button("🗑️", key=f"delete_{i}", help=f"Delete {doc_name}"):
                    # Confirm deletion
                    if 'confirm_delete' not in st.session_state:
                        st.session_state.confirm_delete = doc_name
                        st.rerun()
        
        # Show confirmation dialog if needed
        if 'confirm_delete' in st.session_state:
            doc_to_delete = st.session_state.confirm_delete
            
            st.warning(f"⚠️ Are you sure you want to delete **{doc_to_delete}**?")
            st.caption("This will remove the file and all its chunks from the vector store.")
            
            col_yes, col_no = st.columns(2)
            
            with col_yes:
                if st.button("✅ Yes, Delete", key="confirm_yes", type="primary", use_container_width=True):
                    # Perform deletion
                    try:
                        rag_system = get_rag_system(collection)
                        collection_path = Path(f"./data/documents/{collection}")
                        
                        result = rag_system.delete_document(doc_to_delete, collection_path)
                        
                        if result['success']:
                            st.success(f"✅ {result['message']}")
                            
                            # Clear selection if viewing deleted doc
                            if st.session_state.selected_pdf == doc_to_delete:
                                st.session_state.selected_pdf = None
                            
                            # Clear confirmation state
                            del st.session_state.confirm_delete
                            
                            # Force cache clear and rerun
                            get_rag_system.clear()
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"❌ {result['message']}")
                            del st.session_state.confirm_delete
                            
                    except Exception as e:
                        st.error(f"❌ Error deleting document: {e}")
                        del st.session_state.confirm_delete
            
            with col_no:
                if st.button("❌ Cancel", key="confirm_no", use_container_width=True):
                    del st.session_state.confirm_delete
                    st.rerun()
    
    with col2:
        if st.session_state.selected_pdf:
            st.markdown(f"#### Viewing: {st.session_state.selected_pdf}")
            
            # Try to find PDF in collection folder
            collection_pdf_path = Path(f"./data/documents/{collection}/{st.session_state.selected_pdf}")
            
            if collection_pdf_path.exists():
                display_pdf(str(collection_pdf_path))
            else:
                st.warning(f"PDF file not found in data/documents/{collection}/")
                st.info("The document chunks are in the system, but the original PDF file may have been moved or deleted.")
        else:
            st.info("👈 Select a document from the list to view it")


def show_statistics_view():
    """Show system statistics."""
    st.markdown("### 📊 System Statistics")
    
    collection = st.session_state.current_collection
    
    # Collection stats
    st.markdown(f"#### Collection: `{collection}`")
    metadata = get_collection_metadata(collection)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Documents", metadata['documents'])
    with col2:
        st.metric("📦 Chunks", metadata['chunks'])
    with col3:
        # Calculate avg chunks per doc
        avg_chunks = metadata['chunks'] / metadata['documents'] if metadata['documents'] > 0 else 0
        st.metric("📊 Avg Chunks/Doc", f"{avg_chunks:.1f}")
    
    # Document list
    if metadata['files']:
        st.markdown("#### 📋 Documents in Collection")
        for doc in metadata['files']:
            st.markdown(f"- {doc}")
    
    st.markdown("---")
    
    # All collections overview
    st.markdown("#### 🗂️ All Collections")
    
    collections = get_collection_list()
    if collections:
        for col_name in collections:
            col_meta = get_collection_metadata(col_name)
            folder_path = Path(f"./data/documents/{col_name}")
            folder_exists = folder_path.exists()
            
            # Determine status
            if col_meta['documents'] > 0 and col_meta['indexed'] == 0:
                status = "⚠️ Not indexed"
            elif col_meta['documents'] > 0 and col_meta['indexed'] > 0:
                status = "✅ Indexed"
            else:
                status = "📂 Empty"
            
            st.markdown(f"""
            <div class="collection-card">
                <strong>📁 {col_name}</strong> {status}<br>
                PDFs: {col_meta['documents']} | Chunks: {col_meta['chunks']}<br>
                <small>Folder: data/documents/{col_name}/</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System info
    st.markdown("#### ⚙️ System Configuration")
    try:
        # Use current collection for stats
        current_collection = st.session_state.current_collection
        rag_system = get_rag_system(current_collection)
        stats = rag_system.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Embedding Model:**  
            `{stats['pipeline']['embedding_model']}`
            
            **LLM Model:**  
            `{stats['pipeline']['llm_model']}`
            """)
        
        with col2:
            st.markdown(f"""
            **Total Queries:**  
            {stats['pipeline']['total_queries']}
            
            **Avg Confidence:**  
            {stats['pipeline']['avg_confidence']:.1%}
            """)
    except:
        st.info("System information unavailable")


if __name__ == "__main__":
    main()

