"""
Vector store management using ChromaDB.
"""
import os
import uuid
import numpy as np
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document

from src.rag.config import VectorStoreConfig
from src.rag.utils import get_logger


logger = get_logger(__name__)


class VectorStore:
    """Manages vector database operations with ChromaDB."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.client = None
        self.collection = None
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.config.persist_directory, exist_ok=True)
            
            # Initialize persistent client
            self.client = chromadb.PersistentClient(
                path=self.config.persist_directory
            )
            
            # Get or create collection with explicit distance function
            # For ChromaDB, use 'cosine' for normalized embeddings
            try:
                self.collection = self.client.get_collection(
                    name=self.config.collection_name
                )
            except:
                # Create new collection with cosine distance
                self.collection = self.client.create_collection(
                    name=self.config.collection_name,
                    metadata={"hnsw:space": "cosine"}  # Use cosine distance
                )
            
            logger.info(f"Vector store initialized: {self.config.collection_name}")
            logger.info(f"Existing documents in collection: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(
        self, 
        documents: List[Document], 
        embeddings: np.ndarray,
        batch_size: int = 100
    ) -> List[str]:
        """
        Add documents with embeddings to the vector store.
        
        Args:
            documents: List of Document objects
            embeddings: Numpy array of embeddings
            batch_size: Batch size for adding documents
            
        Returns:
            List of document IDs
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Number of documents ({len(documents)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )
        
        logger.info(f"Adding {len(documents)} documents to vector store...")
        
        all_ids = []
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            ids = []
            metadatas = []
            document_texts = []
            embeddings_list = []
            
            for j, (doc, embedding) in enumerate(zip(batch_docs, batch_embeddings)):
                # Generate unique ID
                doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i+j}"
                ids.append(doc_id)
                
                # Prepare metadata (ChromaDB requires dict with simple types)
                metadata = self._prepare_metadata(doc.metadata, i + j)
                metadatas.append(metadata)
                
                # Document text
                document_texts.append(doc.page_content)
                
                # Embedding as list
                embeddings_list.append(embedding.tolist())
            
            try:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings_list,
                    metadatas=metadatas,
                    documents=document_texts
                )
                all_ids.extend(ids)
                
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size}: {e}")
                raise
        
        logger.info(f"Successfully added {len(documents)} documents")
        logger.info(f"Total documents in collection: {self.collection.count()}")
        
        return all_ids
    
    def _prepare_metadata(self, metadata: dict, doc_index: int) -> dict:
        """
        Prepare metadata for ChromaDB (only simple types allowed).
        
        Args:
            metadata: Original metadata
            doc_index: Document index
            
        Returns:
            Cleaned metadata dictionary
        """
        clean_metadata = {}
        
        for key, value in metadata.items():
            # Convert to simple types
            if isinstance(value, (str, int, float, bool)):
                clean_metadata[key] = value
            elif isinstance(value, (list, tuple)):
                clean_metadata[key] = str(value)
            else:
                clean_metadata[key] = str(value)
        
        # Add index
        clean_metadata['doc_index'] = doc_index
        
        return clean_metadata
    
    def query(
        self, 
        query_embeddings: np.ndarray,
        n_results: int = 10,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store.
        
        Args:
            query_embeddings: Query embedding(s)
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document content filter
            
        Returns:
            Query results dictionary
        """
        try:
            # Ensure query_embeddings is 2D
            if query_embeddings.ndim == 1:
                query_embeddings = query_embeddings.reshape(1, -1)
            
            results = self.collection.query(
                query_embeddings=query_embeddings.tolist(),
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            raise
    
    def delete_documents(self, ids: List[str]):
        """
        Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.config.collection_name)
            self.collection = self.client.create_collection(
                name=self.config.collection_name,
                metadata={
                    "description": "RAG document embeddings",
                    "distance_metric": self.config.distance_metric
                }
            )
            logger.info("Collection cleared")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise
    
    def get_collection_stats(self) -> dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        return {
            'name': self.config.collection_name,
            'count': self.collection.count(),
            'persist_directory': self.config.persist_directory
        }
    
    def document_exists(self, doc_id: str) -> bool:
        """
        Check if a document exists in the collection.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if document exists
        """
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result['ids']) > 0
        except:
            return False
    
    def delete_by_source(self, source_filename: str) -> int:
        """
        Delete all document chunks from a specific source file.
        
        Args:
            source_filename: Name of the source file (e.g., 'document.pdf')
            
        Returns:
            Number of chunks deleted
        """
        try:
            logger.info(f"Attempting to delete chunks for: {source_filename}")
            logger.info(f"Collection: {self.config.collection_name}, Total chunks: {self.collection.count()}")
            
            # First, get a sample to see what keys are being used
            sample = self.collection.get(limit=1, include=['metadatas'])
            if sample['metadatas']:
                logger.info(f"Sample metadata keys: {list(sample['metadatas'][0].keys())}")
            
            # Try different metadata key variations
            chunk_ids = []
            
            # Try 1: source_file
            try:
                results = self.collection.get(
                    where={"source_file": source_filename},
                    include=['metadatas']
                )
                if results['ids']:
                    chunk_ids = results['ids']
                    logger.info(f"Found {len(chunk_ids)} chunks using 'source_file' key")
            except Exception as e:
                logger.warning(f"Query with 'source_file' failed: {e}")
            
            # Try 2: source (if not found yet)
            if not chunk_ids:
                try:
                    results = self.collection.get(
                        where={"source": source_filename},
                        include=['metadatas']
                    )
                    if results['ids']:
                        chunk_ids = results['ids']
                        logger.info(f"Found {len(chunk_ids)} chunks using 'source' key")
                except Exception as e:
                    logger.warning(f"Query with 'source' failed: {e}")
            
            # Try 3: file_path contains filename (fallback)
            if not chunk_ids:
                try:
                    # Get all and filter manually
                    all_results = self.collection.get(include=['metadatas'])
                    for i, metadata in enumerate(all_results['metadatas']):
                        # Check all metadata values for the filename
                        for key, value in metadata.items():
                            if isinstance(value, str) and source_filename in value:
                                chunk_ids.append(all_results['ids'][i])
                                break
                    
                    if chunk_ids:
                        logger.info(f"Found {len(chunk_ids)} chunks by searching metadata values")
                except Exception as e:
                    logger.warning(f"Manual search failed: {e}")
            
            if not chunk_ids:
                logger.error(f"No chunks found for: {source_filename}")
                logger.error("Listing all unique sources in collection:")
                
                # Show what sources exist
                all_results = self.collection.get(include=['metadatas'])
                sources = set()
                for metadata in all_results['metadatas']:
                    for key in ['source_file', 'source', 'file_path']:
                        if key in metadata:
                            sources.add(f"{key}={metadata[key]}")
                
                for source in sorted(sources):
                    logger.error(f"  - {source}")
                
                return 0
            
            # Delete the documents
            self.collection.delete(ids=chunk_ids)
            
            logger.info(f"Successfully deleted {len(chunk_ids)} chunks from: {source_filename}")
            return len(chunk_ids)
            
        except Exception as e:
            logger.error(f"Error deleting by source {source_filename}: {e}")
            raise
    
    def list_sources(self) -> List[Dict[str, Any]]:
        """
        List all unique source documents in the collection with their chunk counts.
        
        Returns:
            List of dictionaries with source info
        """
        try:
            # Get all documents with metadata
            results = self.collection.get(include=['metadatas'])
            
            if not results['metadatas']:
                return []
            
            # Count chunks by source (try both 'source_file' and 'source' keys)
            source_counts = {}
            for metadata in results['metadatas']:
                # Try source_file first (newer format), fallback to source
                source = metadata.get('source_file') or metadata.get('source', 'unknown')
                if source in source_counts:
                    source_counts[source] += 1
                else:
                    source_counts[source] = 1
            
            # Format as list
            sources = [
                {'source': source, 'chunks': count}
                for source, count in sorted(source_counts.items())
            ]
            
            return sources
            
        except Exception as e:
            logger.error(f"Error listing sources: {e}")
            return []