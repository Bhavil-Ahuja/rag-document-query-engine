"""
Embedding generation and management module.
"""
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

from src.rag.config import EmbeddingConfig
from src.rag.utils import get_logger


logger = get_logger(__name__)


class EmbeddingManager:
    """Manages embedding model and generation."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self.dimension = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.config.model_name}")
            self.model = SentenceTransformer(self.config.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Error loading model {self.config.model_name}: {e}")
            raise
    
    def generate_embeddings(
        self, 
        texts: Union[str, List[str]], 
        show_progress: bool = True,
        normalize: bool = None
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings (uses config default if None)
            
        Returns:
            Numpy array of embeddings
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        # Handle single string
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            logger.warning("Empty text list provided")
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} text(s)...")
        
        try:
            # Generate embeddings - always normalize for BGE models
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True  # Always normalize for consistent similarity
            )
            
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        return self.generate_embeddings([query], show_progress=False)[0]
    
    def compute_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        # Normalize if not already
        norm1 = embedding1 / (np.linalg.norm(embedding1) + 1e-10)
        norm2 = embedding2 / (np.linalg.norm(embedding2) + 1e-10)
        
        similarity = np.dot(norm1, norm2)
        return float(similarity)
    
    def batch_compute_similarity(
        self, 
        query_embedding: np.ndarray, 
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity between query and multiple documents.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: Array of document embeddings
            
        Returns:
            Array of similarity scores
        """
        # Normalize
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        doc_norms = document_embeddings / (np.linalg.norm(document_embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Compute dot product
        similarities = np.dot(doc_norms, query_norm)
        return similarities
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.config.model_name,
            'dimension': self.dimension,
            'batch_size': self.config.batch_size,
            'normalize': self.config.normalize
        }