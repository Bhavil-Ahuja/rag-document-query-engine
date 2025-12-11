"""
Advanced retrieval module with hybrid search and re-ranking.
"""
import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from src.rag.config import RetrievalConfig
from src.storage.vector_store import VectorStore
from src.ingestion.embedding_manager import EmbeddingManager
from src.rag.utils import get_logger


logger = get_logger(__name__)


class BM25Retriever:
    """BM25 keyword-based retriever."""
    
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []
    
    def build_index(self, documents: List[Document]):
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of documents
        """
        logger.info(f"Building BM25 index for {len(documents)} documents...")
        self.documents = documents
        self.tokenized_docs = [
            doc.page_content.lower().split() 
            for doc in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        logger.info("BM25 index built successfully")
    
    def search(self, query: str, top_k: int = 10) -> List[tuple]:
        """
        Search using BM25.
        
        Args:
            query: Query string
            top_k: Number of results
            
        Returns:
            List of (doc_index, score) tuples
        """
        if not self.bm25:
            logger.warning("BM25 index not built")
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(idx, scores[idx]) for idx in top_indices]
        
        return results


class ReRanker:
    """Cross-encoder based re-ranker."""
    
    def __init__(self, model_name: str):
        logger.info(f"Loading re-ranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Re-ranker loaded successfully")
    
    def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents using cross-encoder.
        
        Args:
            query: Query string
            documents: List of document dictionaries
            top_k: Number of top results to return
            
        Returns:
            Re-ranked documents
        """
        if not documents:
            return []
        
        logger.info(f"Re-ranking {len(documents)} documents...")
        
        # Prepare pairs
        pairs = [[query, doc['content']] for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Add re-rank scores
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
            # Combine with original similarity score (70% original, 30% rerank)
            doc['combined_score'] = (
                0.7 * doc.get('similarity_score', 0) + 
                0.3 * float(score)
            )
        
        # Sort by combined score
        documents.sort(key=lambda x: x['combined_score'], reverse=True)
        
        logger.info(f"Re-ranking complete. Top score: {documents[0]['combined_score']:.3f}")
        
        return documents[:top_k]


class HybridRetriever:
    """Hybrid retriever combining semantic and keyword search."""
    
    def __init__(
        self, 
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
        config: RetrievalConfig
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.config = config
        
        # Initialize BM25
        self.bm25_retriever = BM25Retriever()
        
        # Initialize re-ranker if enabled
        self.reranker = None
        if config.use_reranker:
            self.reranker = ReRanker(config.reranker_model)
    
    def build_bm25_index(self, documents: List[Document]):
        """
        Build BM25 index.
        
        Args:
            documents: List of documents
        """
        self.bm25_retriever.build_index(documents)
    
    def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        use_hybrid: bool = True,
        hybrid_alpha: Optional[float] = None,  # Override hybrid weight
        use_reranker: Optional[bool] = None  # Override reranker usage
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid search.
        
        Combines:
        - Dense retrieval (semantic/embedding similarity)
        - Sparse retrieval (BM25 keyword matching)
        - Re-ranking (cross-encoder)
        
        Args:
            query: Query string
            top_k: Number of results (uses config default if None)
            score_threshold: Minimum similarity score (uses config default if None)
            use_hybrid: Whether to use hybrid search (BM25 + semantic)
            
        Returns:
            List of retrieved documents with metadata
        """
        top_k = top_k or self.config.top_k
        score_threshold = score_threshold or self.config.min_similarity_score
        hybrid_alpha = hybrid_alpha if hybrid_alpha is not None else self.config.hybrid_alpha
        
        logger.info(f"Retrieving documents for query: '{query}'")
        logger.info(f"Top K: {top_k}, Score threshold: {score_threshold}")
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_query_embedding(query)
        
        # 1. SEMANTIC SEARCH (Dense Retrieval)
        try:
            semantic_results = self.vector_store.query(
                query_embeddings=query_embedding,
                n_results=top_k * 2  # Get more for re-ranking
            )
        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return []
        
        # Parse semantic results
        retrieved_docs = self._parse_results(semantic_results, score_threshold)
        
        if not retrieved_docs:
            logger.warning("No documents found above threshold")
            return []
        
        # 2. HYBRID SEARCH: Combine with BM25 if enabled
        if use_hybrid and self.bm25_retriever.bm25 is not None:
            logger.info("Using hybrid search (semantic + BM25)")
            retrieved_docs = self._hybrid_search(
                query=query,
                semantic_docs=retrieved_docs,
                top_k=top_k * 2,
                alpha=hybrid_alpha  # Use the passed parameter, not config default!
            )
        else:
            logger.info("Using semantic search only")
        
        # 3. Ensure document diversity (prevents excessive overlap issues)
        retrieved_docs = self._ensure_document_diversity(
            retrieved_docs, 
            max_chunks_per_doc=self.config.max_chunks_per_doc
        )
        
        # 4. Re-rank if enabled
        should_rerank = use_reranker if use_reranker is not None else self.config.use_reranker
        if should_rerank and self.reranker and len(retrieved_docs) > 1:
            logger.info("Re-ranking enabled")
            retrieved_docs = self.reranker.rerank(
                query, 
                retrieved_docs, 
                top_k=self.config.reranker_top_k
            )
        else:
            if not should_rerank:
                logger.info("Re-ranking disabled for this query type")
            retrieved_docs = retrieved_docs[:top_k]
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        return retrieved_docs
    
    def _hybrid_search(
        self,
        query: str,
        semantic_docs: List[Dict[str, Any]],
        top_k: int,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Combine semantic search with BM25 keyword search.
        
        Hybrid score = alpha * semantic_score + (1 - alpha) * bm25_score
        
        Args:
            query: Query string
            semantic_docs: Results from semantic search
            top_k: Number of results
            alpha: Weight for semantic vs keyword (1.0 = pure semantic, 0.0 = pure BM25)
            
        Returns:
            Combined and re-scored documents
        """
        # Get BM25 results
        bm25_results = self.bm25_retriever.search(query, top_k=len(self.bm25_retriever.documents))
        
        if not bm25_results:
            logger.warning("BM25 search returned no results")
            return semantic_docs
        
        # Normalize BM25 scores to [0, 1]
        bm25_scores_dict = {}
        max_bm25_score = max(score for _, score in bm25_results) if bm25_results else 1.0
        
        for idx, score in bm25_results:
            if idx < len(self.bm25_retriever.documents):
                doc = self.bm25_retriever.documents[idx]
                normalized_score = score / max_bm25_score if max_bm25_score > 0 else 0
                # Use document content as key for matching
                bm25_scores_dict[doc.page_content] = normalized_score
        
        # Combine scores
        for doc in semantic_docs:
            content = doc['content']
            semantic_score = doc['similarity_score']
            bm25_score = bm25_scores_dict.get(content, 0.0)
            
            # Hybrid score: weighted combination
            hybrid_score = alpha * semantic_score + (1 - alpha) * bm25_score
            
            doc['bm25_score'] = bm25_score
            doc['semantic_score'] = semantic_score
            doc['hybrid_score'] = hybrid_score
            doc['similarity_score'] = hybrid_score  # Update main score
        
        # Sort by hybrid score
        semantic_docs.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        logger.info(f"Hybrid search complete. Alpha={alpha} (semantic weight)")
        logger.info(f"Top result: semantic={semantic_docs[0]['semantic_score']:.3f}, "
                   f"BM25={semantic_docs[0]['bm25_score']:.3f}, "
                   f"hybrid={semantic_docs[0]['hybrid_score']:.3f}")
        
        return semantic_docs[:top_k]
    
    def _ensure_document_diversity(
        self,
        documents: List[Dict[str, Any]],
        max_chunks_per_doc: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Ensure diversity by limiting chunks per source document.
        
        This prevents high chunk overlap from causing all retrieved
        chunks to come from the same document.
        
        For multi-document collections, this ensures the LLM sees
        information from multiple relevant documents, not just one.
        
        Args:
            documents: List of retrieved documents
            max_chunks_per_doc: Maximum chunks allowed from same source
            
        Returns:
            Filtered list with enforced diversity
        """
        if not documents:
            return []
        
        # Group by source document
        doc_groups = {}
        for doc in documents:
            source = doc['metadata'].get('source_file', 'unknown')
            if source not in doc_groups:
                doc_groups[source] = []
            doc_groups[source].append(doc)
        
        # Log document distribution before filtering
        logger.info(f"Retrieved chunks from {len(doc_groups)} document(s):")
        for source, chunks in doc_groups.items():
            logger.info(f"  - {source}: {len(chunks)} chunks")
        
        # Select top chunks from each document (round-robin for diversity)
        diverse_docs = []
        doc_counts = {source: 0 for source in doc_groups}
        
        # Round-robin selection
        max_iterations = max_chunks_per_doc
        for iteration in range(max_iterations):
            for source, chunks in doc_groups.items():
                if doc_counts[source] < len(chunks) and doc_counts[source] < max_chunks_per_doc:
                    diverse_docs.append(chunks[doc_counts[source]])
                    doc_counts[source] += 1
        
        if len(diverse_docs) < len(documents):
            removed = len(documents) - len(diverse_docs)
            logger.info(f"Filtered {removed} redundant chunks for diversity")
        
        return diverse_docs
    
    def _parse_results(
        self, 
        results: Dict[str, Any], 
        score_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Parse ChromaDB results into structured format.
        
        Args:
            results: Raw results from ChromaDB
            score_threshold: Minimum score threshold
            
        Returns:
            List of parsed document dictionaries
        """
        retrieved_docs = []
        
        if not results['documents'] or not results['documents'][0]:
            return retrieved_docs
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        ids = results['ids'][0]
        
        for i, (doc_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances)
        ):
            # Convert distance to similarity
            similarity_score = 1 - distance
            
            if similarity_score >= score_threshold:
                retrieved_docs.append({
                    'id': doc_id,
                    'content': document,
                    'metadata': metadata,
                    'similarity_score': similarity_score,
                    'distance': distance,
                    'rank': i + 1
                })
        
        return retrieved_docs
    
    def get_retriever_stats(self) -> dict:
        """
        Get retriever statistics.
        
        Returns:
            Dictionary with retriever stats
        """
        return {
            'top_k': self.config.top_k,
            'min_similarity_score': self.config.min_similarity_score,
            'use_reranker': self.config.use_reranker,
            'reranker_model': self.config.reranker_model if self.config.use_reranker else None,
            'vector_store_docs': self.vector_store.collection.count()
        }