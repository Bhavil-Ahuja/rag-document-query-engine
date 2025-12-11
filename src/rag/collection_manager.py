"""
Collection Manager for Multi-Collection RAG System

Handles multiple collections/folders for organizing documents.
"""

import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from src.ingestion.embedding_manager import EmbeddingManager
from .config import get_default_config
from .utils import get_logger

logger = get_logger(__name__)


class CollectionManager:
    """Manages multiple ChromaDB collections."""
    
    def __init__(self, persist_directory: str = "./data/vector_store"):
        """
        Initialize collection manager.
        
        Args:
            persist_directory: Directory for vector store persistence
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.config = get_default_config()
        self.embedding_manager = EmbeddingManager(self.config.embedding)
        
        logger.info(f"Collection manager initialized with {len(self.list_collections())} collections")
    
    def list_collections(self) -> List[str]:
        """
        List all collections.
        
        Returns:
            List of collection names
        """
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def create_collection(self, name: str) -> bool:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            
        Returns:
            True if successful
        """
        try:
            # Check if exists
            existing = self.list_collections()
            if name in existing:
                logger.warning(f"Collection '{name}' already exists")
                return False
            
            # Create collection with cosine distance
            self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created collection: {name}")
            return True
        
        except Exception as e:
            logger.error(f"Error creating collection '{name}': {e}")
            return False
    
    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            name: Collection name
            
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(name)
            logger.info(f"Deleted collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection '{name}': {e}")
            return False
    
    def get_collection(self, name: str):
        """
        Get a collection.
        
        Args:
            name: Collection name
            
        Returns:
            Collection object
        """
        try:
            return self.client.get_collection(name)
        except Exception as e:
            logger.error(f"Error getting collection '{name}': {e}")
            return None
    
    def get_or_create_collection(self, name: str):
        """
        Get or create a collection.
        
        Args:
            name: Collection name
            
        Returns:
            Collection object
        """
        try:
            return self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(f"Error getting/creating collection '{name}': {e}")
            return None
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[Document],
        show_progress: bool = True
    ) -> int:
        """
        Add documents to a collection.
        
        Args:
            collection_name: Target collection
            documents: List of documents
            show_progress: Whether to show progress
            
        Returns:
            Number of documents added
        """
        if not documents:
            logger.warning("No documents to add")
            return 0
        
        try:
            # Get or create collection
            collection = self.get_or_create_collection(collection_name)
            if not collection:
                return 0
            
            # Generate embeddings
            texts = [doc.page_content for doc in documents]
            embeddings = self.embedding_manager.generate_embeddings(
                texts,
                show_progress=show_progress
            )
            
            # Prepare data
            ids = [f"doc_{collection_name}_{i}" for i in range(len(documents))]
            metadatas = [doc.metadata for doc in documents]
            
            # Add to collection
            collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to collection '{collection_name}'")
            return len(documents)
        
        except Exception as e:
            logger.error(f"Error adding documents to '{collection_name}': {e}")
            return 0
    
    def query_collection(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 10
    ) -> Dict[str, Any]:
        """
        Query a collection.
        
        Args:
            collection_name: Collection to query
            query_text: Query text
            n_results: Number of results
            
        Returns:
            Query results
        """
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
            
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_query_embedding(query_text)
            
            # Query collection
            results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results
            )
            
            return results
        
        except Exception as e:
            logger.error(f"Error querying collection '{collection_name}': {e}")
            return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Args:
            collection_name: Collection name
            
        Returns:
            Statistics dictionary
        """
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return {'count': 0, 'documents': 0, 'files': []}
            
            count = collection.count()
            
            # Get sample to find unique documents
            results = collection.get(limit=1000, include=['metadatas'])
            doc_files = set()
            if results and results.get('metadatas'):
                for metadata in results['metadatas']:
                    if metadata and 'source_file' in metadata:
                        doc_files.add(metadata['source_file'])
            
            return {
                'count': count,
                'documents': len(doc_files),
                'files': sorted(list(doc_files))
            }
        
        except Exception as e:
            logger.error(f"Error getting stats for '{collection_name}': {e}")
            return {'count': 0, 'documents': 0, 'files': []}