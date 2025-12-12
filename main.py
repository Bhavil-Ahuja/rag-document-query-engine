"""
Main entry point for the RAG pipeline.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.rag.config import get_default_config, RAGConfig
from src.ingestion.document_processor import DocumentProcessor
from src.rag.pipeline import RAGPipeline
from src.evaluation.evaluator import RAGEvaluator
from src.rag.utils import get_logger


logger = get_logger(__name__)


class RAGSystem:
    """Complete RAG system orchestrator."""
    
    def __init__(self, config: Optional[RAGConfig] = None, collection_name: Optional[str] = None):
        """
        Initialize RAG system.
        
        Args:
            config: RAG configuration (uses defaults if None)
            collection_name: Override collection name (optional)
        """
        self.config = config or get_default_config()
        
        # Override collection name if provided
        if collection_name:
            self.config.vector_store.collection_name = collection_name
        
        logger.info("=" * 60)
        logger.info("Initializing RAG System")
        logger.info(f"Collection: {self.config.vector_store.collection_name}")
        logger.info("=" * 60)
        
        # Initialize components
        self.document_processor = DocumentProcessor(self.config.chunking)
        self.pipeline = RAGPipeline(self.config)
        self.evaluator = RAGEvaluator() if self.config.enable_evaluation else None
        
        # Build BM25 index for existing documents if hybrid search is enabled
        if self.config.retrieval.use_hybrid_search:
            self._build_bm25_index_from_existing()
        
        logger.info("RAG System ready!")
        logger.info("=" * 60)
    
    def ingest_documents_from_directory(
        self, 
        directory: Path,
        deduplicate: bool = True,
        clean: bool = True
    ) -> int:
        """
        Ingest all documents from a directory.
        
        Args:
            directory: Directory containing documents
            deduplicate: Whether to remove duplicates
            clean: Whether to clean text
            
        Returns:
            Number of documents ingested
        """
        logger.info(f"Processing documents from: {directory}")
        
        # Process documents
        chunks = self.document_processor.process_directory(
            directory,
            deduplicate=deduplicate,
            clean=clean
        )
        
        if not chunks:
            logger.warning("No documents to ingest")
            return 0
        
        # Ingest into pipeline
        count = self.pipeline.ingest_documents(chunks)
        
        return count
    
    def ingest_file(self, file_path: Path) -> int:
        """
        Ingest a single file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Number of chunks ingested
        """
        logger.info(f"Processing file: {file_path}")
        
        chunks = self.document_processor.process_file(file_path)
        
        if not chunks:
            logger.warning("No chunks created from file")
            return 0
        
        count = self.pipeline.ingest_documents(chunks)
        
        return count
    
    def _build_bm25_index_from_existing(self):
        """
        Build BM25 index from existing documents in the vector store.
        This is called on initialization to enable hybrid search for existing documents.
        """
        try:
            # Get all existing documents from vector store
            collection = self.pipeline.vector_store.collection
            result = collection.get(include=["documents", "metadatas"])
            
            if result and result['documents']:
                # Convert to Document objects
                from langchain_core.documents import Document
                documents = [
                    Document(page_content=doc, metadata=meta) 
                    for doc, meta in zip(result['documents'], result['metadatas'])
                ]
                
                # Build BM25 index
                self.pipeline.retriever.build_bm25_index(documents)
                logger.info(f"Built BM25 index for {len(documents)} existing documents")
            else:
                logger.info("No existing documents found - BM25 index will be built on first ingestion")
                
        except Exception as e:
            logger.warning(f"Could not build BM25 index from existing documents: {e}")
            logger.info("BM25 index will be built when new documents are ingested")
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        evaluate: bool = False
    ) -> dict:
        """
        Query the RAG system.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            min_score: Minimum similarity score
            evaluate: Whether to evaluate response
            
        Returns:
            Response dictionary
        """
        # Get response from pipeline
        response = self.pipeline.query(
            question,
            top_k=top_k,
            min_score=min_score,
            return_sources=True,
            return_context=evaluate  # Only get context if evaluating
        )
        
        # Evaluate if requested
        if evaluate and self.evaluator:
            metrics = self.evaluator.evaluate_response(
                query=question,
                answer=response['answer'],
                context=response.get('context', ''),
                sources=response.get('sources', [])
            )
            response['evaluation'] = metrics
        
        return response
    
    def get_stats(self) -> dict:
        """
        Get system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        stats = {
            'pipeline': self.pipeline.get_stats(),
            'retriever': self.pipeline.retriever.get_retriever_stats(),
            'vector_store': self.pipeline.vector_store.get_collection_stats()
        }
        
        if self.evaluator:
            stats['evaluation'] = self.evaluator.get_session_stats()
        
        return stats
    
    def clear_cache(self):
        """Clear query cache."""
        self.pipeline.clear_cache()
    
    def save_evaluation_log(self):
        """Save evaluation log."""
        if self.evaluator:
            self.evaluator.save_session_log()
    
    def delete_document(self, filename: str, collection_path: Path) -> dict:
        """
        Delete a document from both filesystem and vector store.
        
        Args:
            filename: Name of the PDF file to delete
            collection_path: Path to the collection folder
            
        Returns:
            Dictionary with deletion results
        """
        try:
            result = {
                'success': False,
                'file_deleted': False,
                'chunks_deleted': 0,
                'message': ''
            }
            
            # Delete from filesystem
            file_path = collection_path / filename
            if file_path.exists():
                file_path.unlink()
                result['file_deleted'] = True
                logger.info(f"Deleted file: {filename}")
            else:
                logger.warning(f"File not found: {filename}")
            
            # Delete from vector store (all chunks with this source)
            chunks_deleted = self.pipeline.vector_store.delete_by_source(filename)
            result['chunks_deleted'] = chunks_deleted
            
            # Rebuild BM25 index if using hybrid search
            if self.config.retrieval.use_hybrid_search:
                self._build_bm25_index_from_existing()
            
            result['success'] = True
            result['message'] = f"Deleted {filename}: {chunks_deleted} chunks removed"
            
            logger.info(f"Successfully deleted document: {filename}")
            return result
            
        except Exception as e:
            logger.error(f"Error deleting document {filename}: {e}")
            return {
                'success': False,
                'file_deleted': False,
                'chunks_deleted': 0,
                'message': f"Error: {str(e)}"
            }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the vector store.
        
        Returns:
            List of document sources with their chunk counts
        """
        try:
            return self.pipeline.vector_store.list_sources()
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []


def interactive_mode():
    """Interactive terminal mode for querying the RAG system."""
    
    # Initialize system
    print("\n" + "=" * 70)
    print("🚀 RAG System - Interactive Mode")
    print("=" * 70)
    
    # Ingest documents from data/documents/ folder
    documents_dir = Path("./data/documents")
    
    # Check if documents directory exists and has collections
    if documents_dir.exists():
        collections = [d for d in documents_dir.iterdir() if d.is_dir()]
        
        if collections:
            print(f"\n📁 Found {len(collections)} collection(s) in data/documents/")
            
            # Use first collection
            first_collection = collections[0]
            collection_name = first_collection.name
            print(f"📄 Using collection: {collection_name}")
            
            # Initialize RAG system with specific collection
            rag_system = RAGSystem(collection_name=collection_name)
            
            count = rag_system.ingest_documents_from_directory(first_collection)
            print(f"✅ Ingested {count} document chunks from {collection_name}\n")
        else:
            print(f"⚠️  No collections found in data/documents/")
            print("Please create a collection and add PDF files using the web UI.\n")
            return
    else:
        print(f"⚠️  Directory not found: {documents_dir}")
        print("Please run the web UI first to create collections: streamlit run app_advanced.py\n")
        return
    
    # Print help
    print("=" * 70)
    print("Commands:")
    print("  - Type your question and press Enter")
    print("  - 'stats'  : Show system statistics")
    print("  - 'clear'  : Clear query cache")
    print("  - 'history': Show query history")
    print("  - 'help'   : Show this help message")
    print("  - 'quit' or 'exit' : Exit the system")
    print("=" * 70 + "\n")
    
    query_count = 0
    
    # Interactive loop
    while True:
        try:
            # Get user input
            user_input = input("💬 You: ").strip()
            
            # Handle empty input
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Saving evaluation log and exiting...")
                rag_system.save_evaluation_log()
                stats = rag_system.get_stats()
                print(f"\n📊 Session Summary:")
                print(f"   Total queries: {stats['pipeline']['total_queries']}")
                print(f"   Avg confidence: {stats['pipeline']['avg_confidence']:.2%}")
                print(f"   Cache hits: {stats['pipeline']['cache_size']}")
                print("\n✅ Goodbye!\n")
                break
            
            elif user_input.lower() == 'stats':
                print("\n📊 System Statistics:")
                print("-" * 70)
                stats = rag_system.get_stats()
                print(f"Pipeline:")
                print(f"  Total queries: {stats['pipeline']['total_queries']}")
                print(f"  Average confidence: {stats['pipeline']['avg_confidence']:.2%}")
                print(f"  Cache size: {stats['pipeline']['cache_size']} queries")
                print(f"  Documents in vector store: {stats['pipeline']['vector_store_docs']}")
                print(f"\nModels:")
                print(f"  Embedding: {stats['pipeline']['embedding_model']}")
                print(f"  LLM: {stats['pipeline']['llm_model']}")
                print(f"\nRetriever:")
                print(f"  Top K: {stats['retriever']['top_k']}")
                print(f"  Min similarity: {stats['retriever']['min_similarity_score']:.2f}")
                print(f"  Re-ranker: {stats['retriever']['use_reranker']}")
                print()
                continue
            
            elif user_input.lower() == 'clear':
                rag_system.clear_cache()
                print("✅ Cache cleared!\n")
                continue
            
            elif user_input.lower() == 'history':
                history = rag_system.pipeline.history
                if not history:
                    print("📝 No query history yet.\n")
                else:
                    print(f"\n📝 Query History ({len(history)} queries):")
                    print("-" * 70)
                    for i, entry in enumerate(history[-10:], 1):  # Show last 10
                        print(f"{i}. Q: {entry['question']}")
                        print(f"   A: {entry['answer'][:100]}...")
                        print(f"   Confidence: {entry['confidence']:.2%}\n")
                continue
            
            elif user_input.lower() == 'help':
                print("\n" + "=" * 70)
                print("Commands:")
                print("  - Type your question and press Enter")
                print("  - 'stats'  : Show system statistics")
                print("  - 'clear'  : Clear query cache")
                print("  - 'history': Show query history")
                print("  - 'help'   : Show this help message")
                print("  - 'quit' or 'exit' : Exit the system")
                print("=" * 70 + "\n")
                continue
            
            # Process query
            query_count += 1
            print(f"\n🔍 Processing query {query_count}...\n")
            
            response = rag_system.query(user_input, evaluate=True)
            
            # Display answer
            print("🤖 Assistant:")
            print("-" * 70)
            print(response['answer'])
            print("-" * 70)
            
            # Display metadata
            print(f"\n📈 Metadata:")
            print(f"   Confidence: {response['confidence']:.2%}")
            print(f"   Retrieved docs: {response['retrieved_docs']}")
            print(f"   Query time: {response['query_time']:.2f}s")
            
            # Display evaluation if available
            if 'evaluation' in response:
                print(f"\n📊 Quality Scores:")
                print(f"   Faithfulness: {response['evaluation']['faithfulness']:.2%}")
                print(f"   Relevance: {response['evaluation']['relevance']:.2%}")
                print(f"   Overall: {response['evaluation']['overall_score']:.2%}")
            
            # Display sources
            if response['sources']:
                print(f"\n📚 Sources:")
                for i, src in enumerate(response['sources'][:3], 1):
                    print(f"   [{i}] {src['source']} (page {src['page']}, score: {src['score']:.2%})")
            
            # Display retrieved document contents
            if response['sources']:
                print(f"\n📄 Retrieved Document Contents:")
                print("=" * 70)
                for i, src in enumerate(response['sources'], 1):
                    print(f"\n[Document {i}] {src['source']} (Page {src['page']}, Score: {src['score']:.2%})")
                    print("-" * 70)
                    # Display full content
                    content = src.get('content', 'No content available')
                    print(content)
                    print("-" * 70)
                print("=" * 70)
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Type 'quit' to exit gracefully.\n")
            continue
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue


def main():
    """Main entry point - launches interactive mode."""
    interactive_mode()


if __name__ == "__main__":
    main()