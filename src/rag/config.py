"""
Configuration settings for the RAG pipeline.
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str = "BAAI/bge-large-en-v1.5"  # or "all-MiniLM-L6-v2" for faster
    dimension: int = 1024  # 384 for MiniLM, 1024 for bge-large
    batch_size: int = 32
    normalize: bool = True


@dataclass
class VectorStoreConfig:
    """Configuration for vector database."""
    collection_name: str = "pdf_documents"  # Default collection (can be overridden)
    persist_directory: str = "./data/vector_store"
    distance_metric: str = "cosine"


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    chunk_size: int = 400          # Smaller for precision (invoices/bills)
    chunk_overlap: int = 80        # 20% overlap
    separators: List[str] = None
    use_parent_chunks: bool = False  # Enable parent document retrieval
    parent_chunk_size: int = 1200   # Larger parent chunks for context
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    top_k: int = 10
    min_similarity_score: float = 0.3
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5
    
    # Hybrid Search (Dense + Sparse)
    use_hybrid_search: bool = True  # Enable hybrid search (semantic + BM25)
    hybrid_alpha: float = 0.5  # Weight: 1.0 = pure semantic, 0.0 = pure BM25, 0.5 = balanced (better for dates!)
    
    max_chunks_per_doc: int = 3  # Allow more chunks per doc for multi-doc collections
    enable_document_filtering: bool = True  # Filter by document metadata


@dataclass
class LLMConfig:
    """Configuration for LLM."""
    provider: str = "groq"  # groq, openai, anthropic
    model_name: str = "llama-3.1-8b-instant"
    temperature: float = 0.1
    max_tokens: int = 1024
    api_key: str = os.getenv("GROQ_API_KEY", "")


@dataclass
class RAGConfig:
    """Main RAG configuration."""
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    
    # Paths
    data_dir: Path = Path("./data")
    documents_dir: Path = Path("./data/documents")  # Base folder for all collections
    pdf_dir: Path = Path("./data/documents")  # For backwards compatibility
    cache_dir: Path = Path("./data/cache")
    
    # Features
    enable_caching: bool = True
    enable_evaluation: bool = True
    enable_query_optimization: bool = True
    
    # Advanced Query Optimization
    use_hyde: bool = True  # Hypothetical Document Embeddings
    use_stepback: bool = True  # Step-back prompting for context
    use_query2doc: bool = True  # Query-to-document augmentation
    use_multiquery: bool = True  # Generate multiple query variations
    
    # Query-type specific optimizations
    disable_advanced_for_temporal: bool = True  # Temporal queries work better with simple BM25
    
    # Hallucination prevention
    min_confidence_threshold: float = 0.35  # Lowered from 0.5 for better recall
    require_source_citation: bool = True
    
    def __post_init__(self):
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)


def get_default_config() -> RAGConfig:
    """Get default configuration."""
    return RAGConfig(
        embedding=EmbeddingConfig(),
        vector_store=VectorStoreConfig(),
        chunking=ChunkingConfig(),
        retrieval=RetrievalConfig(),
        llm=LLMConfig()
    )