"""
Main RAG pipeline with hallucination prevention and advanced features.
"""
import time
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.documents import Document

from .config import RAGConfig
from src.ingestion.embedding_manager import EmbeddingManager
from src.storage.vector_store import VectorStore
from src.retrieval.retriever import HybridRetriever
from src.retrieval.query_optimizer import QueryOptimizer, QueryRouter
from .utils import get_logger, format_sources, compute_query_hash


logger = get_logger(__name__)


class RAGPipeline:
    """Production-grade RAG pipeline with hallucination prevention."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # Initialize components
        logger.info("Initializing RAG pipeline...")
        
        self.embedding_manager = EmbeddingManager(config.embedding)
        self.vector_store = VectorStore(config.vector_store)
        self.retriever = HybridRetriever(
            self.vector_store, 
            self.embedding_manager, 
            config.retrieval
        )
        
        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=config.llm.api_key,
            model_name=config.llm.model_name,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens
        )
        
        # Optional components
        self.query_optimizer = None
        if config.enable_query_optimization:
            self.query_optimizer = QueryOptimizer(config.llm)
            self.query_router = QueryRouter()
        
        # Cache
        self.cache = {} if config.enable_caching else None
        
        # Query history
        self.history = []
        
        logger.info("RAG pipeline initialized successfully")
    
    def ingest_documents(
        self, 
        documents: List[Document],
        show_progress: bool = True
    ) -> int:
        """
        Ingest documents into the vector store.
        
        Args:
            documents: List of document chunks
            show_progress: Whether to show progress
            
        Returns:
            Number of documents ingested
        """
        if not documents:
            logger.warning("No documents to ingest")
            return 0
        
        logger.info(f"Ingesting {len(documents)} documents...")
        
        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_manager.generate_embeddings(
            texts, 
            show_progress=show_progress
        )
        
        # Add to vector store
        doc_ids = self.vector_store.add_documents(documents, embeddings)
        
        # Build BM25 index for hybrid search
        self.retriever.build_bm25_index(documents)
        
        logger.info(f"Successfully ingested {len(doc_ids)} documents")
        
        return len(doc_ids)
    
    def query(
        self, 
        question: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        return_sources: bool = True,
        return_context: bool = False
    ) -> Dict[str, Any]:
        """
        Query the RAG system with hallucination prevention.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            min_score: Minimum similarity score
            return_sources: Whether to return source citations
            return_context: Whether to return retrieved context
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        logger.info(f"Processing query: {question}")
        
        # Check cache
        if self.cache is not None:
            cache_key = compute_query_hash(question, {'top_k': top_k, 'min_score': min_score})
            if cache_key in self.cache:
                logger.info("Cache hit!")
                return self.cache[cache_key]
        
        # Optimize query if enabled
        doc_hints = None
        all_results = []
        
        if self.query_optimizer:
            optimization = self.query_optimizer.optimize_query(
                question,
                use_hyde=self.config.use_hyde,
                use_stepback=self.config.use_stepback,
                use_query2doc=self.config.use_query2doc,
                use_multiquery=self.config.use_multiquery
            )
            query_type = optimization['type']
            
            # For temporal queries, disable advanced optimizations (simple BM25 works better!)
            if query_type == 'temporal' and self.config.disable_advanced_for_temporal:
                logger.info("Temporal query: Disabling advanced optimizations for direct BM25 matching")
                
                # Extract entity from query (e.g., "Decathlon" from "When did I purchase from Decathlon?")
                # Add explicit date/time keywords to help BM25
                temporal_query = f"{question} date time"
                
                optimization = {
                    'original': question,
                    'type': 'temporal',
                    'variations': [temporal_query],  # Add date/time keywords!
                    'sub_queries': None,
                    'hyde_document': None,
                    'stepback_question': None,
                    'query2doc': None,
                    'strategy': 'simple_temporal'
                }
                logger.info(f"Temporal query augmented to: '{temporal_query}'")
            
            # Extract document references from query
            doc_hints = self.query_optimizer.extract_document_reference(question)
            
            # Get routing parameters
            route_params = self.query_router.route(query_type)
            top_k = top_k or route_params['top_k']
            
            # Use config defaults if not specified
            top_k = top_k or self.config.retrieval.top_k
            min_score = min_score or self.config.retrieval.min_similarity_score
            
            # For temporal queries, use MORE BM25 weight and disable re-ranker (dates are keywords!)
            hybrid_alpha_override = None
            use_reranker_override = route_params['rerank']  # Use routing decision
            if query_type == 'temporal':
                hybrid_alpha_override = 0.1  # 10% semantic, 90% BM25 for better keyword matching!
                use_reranker_override = False  # Disable re-ranker for temporal queries
                logger.info("Temporal query detected: hybrid_alpha=0.1 (90% BM25), reranker=False")
            
            # HYDE: Retrieve using hypothetical document
            if optimization.get('hyde_document'):
                logger.info("Retrieving with Hyde document")
                hyde_results = self.retriever.retrieve(
                    optimization['hyde_document'],
                    top_k=top_k,
                    score_threshold=min_score,
                    use_hybrid=self.config.retrieval.use_hybrid_search,
                    hybrid_alpha=hybrid_alpha_override,
                    use_reranker=use_reranker_override
                )
                all_results.extend(hyde_results)
            
            # STEPBACK: Retrieve using both original and step-back question
            if optimization.get('stepback_question'):
                logger.info("Retrieving with step-back question")
                stepback_results = self.retriever.retrieve(
                    optimization['stepback_question'],
                    top_k=top_k // 2,  # Half for stepback
                    score_threshold=min_score,
                    use_hybrid=self.config.retrieval.use_hybrid_search,
                    hybrid_alpha=hybrid_alpha_override,
                    use_reranker=use_reranker_override
                )
                all_results.extend(stepback_results)
            
            # MULTIQUERY: Retrieve with all query variations
            if self.config.use_multiquery and optimization.get('variations'):
                logger.info(f"Retrieving with {len(optimization['variations'])} query variations")
                for variation in optimization['variations']:
                    var_results = self.retriever.retrieve(
                        variation,
                        top_k=top_k // len(optimization['variations']),
                        score_threshold=min_score,
                        use_hybrid=self.config.retrieval.use_hybrid_search,
                        hybrid_alpha=hybrid_alpha_override,
                        use_reranker=use_reranker_override
                    )
                    all_results.extend(var_results)
            
            # DECOMPOSED: Retrieve for each sub-query and merge
            if optimization.get('sub_queries'):
                logger.info(f"Retrieving for {len(optimization['sub_queries'])} sub-queries")
                for sub_q in optimization['sub_queries']:
                    sub_results = self.retriever.retrieve(
                        sub_q,
                        top_k=top_k // len(optimization['sub_queries']),
                        score_threshold=min_score,
                        use_hybrid=self.config.retrieval.use_hybrid_search
                    )
                    all_results.extend(sub_results)
        
        # Use config defaults if not specified
        top_k = top_k or self.config.retrieval.top_k
        min_score = min_score or self.config.retrieval.min_similarity_score
        
        # If no advanced techniques were used, do standard retrieval
        if not all_results:
            hybrid_alpha_to_use = hybrid_alpha_override if self.query_optimizer else None
            use_reranker_to_use = use_reranker_override if self.query_optimizer else None
            all_results = self.retriever.retrieve(
                question,
                top_k=top_k,
                score_threshold=min_score,
                use_hybrid=self.config.retrieval.use_hybrid_search,
                hybrid_alpha=hybrid_alpha_to_use,
                use_reranker=use_reranker_to_use
            )
        else:
            # Deduplicate and re-rank combined results
            all_results = self._deduplicate_and_rerank(all_results, top_k)
        
        results = all_results
        
        # Filter by document if specific document mentioned in query
        if doc_hints and doc_hints.get('has_specific_doc') and results:
            results = self._filter_by_document_hints(results, doc_hints)
        
        # Check if we have reliable results
        if not results:
            response = {
                'answer': "I don't have enough reliable information to answer this question based on the available documents.",
                'sources': [],
                'confidence': 0.0,
                'query_time': time.time() - start_time,
                'retrieved_docs': 0
            }
            self._add_to_history(question, response)
            return response
        
        # Calculate confidence
        confidence = max([doc['similarity_score'] for doc in results])
        
        # Check confidence threshold
        if confidence < self.config.min_confidence_threshold:
            response = {
                'answer': f"I found some potentially relevant information, but I'm not confident enough (confidence: {confidence:.2f}) to provide a reliable answer.",
                'sources': self._format_sources(results) if return_sources else [],
                'confidence': confidence,
                'query_time': time.time() - start_time,
                'retrieved_docs': len(results)
            }
            self._add_to_history(question, response)
            return response
        
        # Prepare context with source labels for each chunk
        context_parts = []
        for i, doc in enumerate(results, 1):
            source = doc['metadata'].get('source_file', 'unknown')
            page = doc['metadata'].get('page', 'unknown')
            content = doc['content']
            context_parts.append(f"[Chunk {i} from {source}, Page {page}]:\n{content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer with hallucination prevention
        answer = self._generate_answer(question, context, results)
        
        # Prepare response
        response = {
            'answer': answer,
            'sources': self._format_sources(results) if return_sources else [],
            'confidence': confidence,
            'query_time': time.time() - start_time,
            'retrieved_docs': len(results)
        }
        
        if return_context:
            response['context'] = context
        
        # Add to history
        self._add_to_history(question, response)
        
        # Cache result
        if self.cache is not None:
            self.cache[cache_key] = response
        
        logger.info(f"Query completed in {response['query_time']:.2f}s")
        
        return response
    
    def _deduplicate_and_rerank(
        self,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate and re-rank results from multiple queries.
        
        When using MultiQuery, Hyde, or StepBack, we get results from
        multiple retrieval operations. This method:
        1. Removes duplicates based on content hash
        2. Aggregates scores for duplicates
        3. Re-ranks by aggregated score
        4. Returns top_k results
        
        Args:
            results: List of retrieved documents (may have duplicates)
            top_k: Number of top results to return
            
        Returns:
            Deduplicated and re-ranked results
        """
        if not results:
            return []
        
        # Track unique documents by content
        seen_content = {}
        
        for doc in results:
            content = doc['content']
            
            if content in seen_content:
                # Aggregate scores (take max)
                existing_score = seen_content[content]['similarity_score']
                new_score = doc['similarity_score']
                seen_content[content]['similarity_score'] = max(existing_score, new_score)
                # Increment retrieval count
                seen_content[content]['retrieval_count'] = seen_content[content].get('retrieval_count', 1) + 1
            else:
                # First time seeing this content
                doc['retrieval_count'] = 1
                seen_content[content] = doc
        
        # Convert back to list and sort by score * retrieval_count
        # Documents retrieved multiple times are likely more relevant
        unique_results = list(seen_content.values())
        unique_results.sort(
            key=lambda x: x['similarity_score'] * (1 + 0.1 * x['retrieval_count']),
            reverse=True
        )
        
        logger.info(f"Deduplicated {len(results)} → {len(unique_results)} unique documents")
        
        return unique_results[:top_k]
    
    def _filter_by_document_hints(
        self, 
        results: List[Dict[str, Any]], 
        doc_hints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter results based on document references in query.
        
        Args:
            results: Retrieved documents
            doc_hints: Document filtering hints from query
            
        Returns:
            Filtered results
        """
        if not doc_hints or not doc_hints.get('keywords'):
            return results
        
        keywords = doc_hints['keywords']
        months = doc_hints.get('months', [])
        
        filtered = []
        for doc in results:
            source_file = doc['metadata'].get('source_file', '').lower()
            
            # Check if any keyword matches the source file
            keyword_match = any(kw in source_file for kw in keywords)
            month_match = any(month in source_file for month in months)
            
            if keyword_match or month_match:
                filtered.append(doc)
        
        # If filtering removed everything, return original results
        if not filtered:
            logger.info("Document filtering too restrictive, using all results")
            return results
        
        logger.info(f"Filtered to {len(filtered)} chunks from specific document(s)")
        return filtered
    
    def _generate_answer(
        self, 
        question: str, 
        context: str, 
        sources: List[Dict[str, Any]]
    ) -> str:
        """
        Generate answer with hallucination prevention.
        
        Args:
            question: User question
            context: Retrieved context
            sources: Source documents
            
        Returns:
            Generated answer
        """
        # Get unique source documents to know if multi-doc
        source_files = set(src['metadata'].get('source_file', 'unknown') for src in sources)
        is_multi_doc = len(source_files) > 1
        
        # Build explicit list of available sources
        source_list = "\n".join([f"- {filename}" for filename in sorted(source_files)])
        
        # Build prompt with strict grounding instructions
        multi_doc_instruction = ""
        if is_multi_doc:
            multi_doc_instruction = f"""
⚠️ MULTIPLE DOCUMENTS DETECTED:
The context below contains excerpts from {len(source_files)} different documents.
When answering, you MUST specify which document each piece of information comes from.
"""
        
        prompt = f"""You are a precise and accurate assistant. Answer questions based STRICTLY on the provided context.

📋 AVAILABLE SOURCES (These are the ONLY documents you can reference):
{source_list}

⚠️ CRITICAL RULES - FOLLOW STRICTLY:
1. Read the context CAREFULLY to identify which document contains the specific information needed
2. ONLY cite the document that ACTUALLY contains the answer to the question
3. DO NOT mix up information between documents
4. Each chunk in the context comes from a specific document - match your answer to the correct document
5. For each piece of information, mention the EXACT document name it came from (e.g., "According to Balls.pdf...")
6. If multiple documents have relevant info, list each piece with its correct source
7. If the context doesn't contain the answer, say "The provided documents don't contain this information"
8. NEVER attribute information to the wrong document
{multi_doc_instruction}
CONTEXT FROM AVAILABLE SOURCES (each chunk is labeled with its source):
{context}

QUESTION: {question}

ANSWER (carefully match each fact to its correct source document):"""
        
        try:
            response = self.llm.invoke([prompt])
            answer = response.content.strip()
            
            # Validate: Check if LLM cited any documents not in sources
            invalid_citations = self._validate_citations(answer, source_files)
            if invalid_citations:
                logger.warning(f"LLM hallucinated citations: {invalid_citations}")
                # Add warning to answer
                answer += f"\n\n⚠️ Note: The answer above may have incorrectly referenced documents not in the provided sources."
            
            # Note: Sources are shown in the UI's "View Details" section
            # No need to append them to the answer text to avoid duplication
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I encountered an error while generating the answer. Please try again."
    
    def _validate_citations(self, answer: str, valid_sources: set) -> List[str]:
        """
        Check if LLM cited any documents not in the provided sources.
        
        Args:
            answer: LLM's answer
            valid_sources: Set of valid source filenames
            
        Returns:
            List of invalid citations found
        """
        import re
        
        invalid = []
        answer_lower = answer.lower()
        
        # Common document name patterns in answers
        patterns = [
            r'according to (\w+\.pdf)',
            r'in (\w+\.pdf)',
            r'from (\w+\.pdf)',
            r'in the (\w+) (?:invoice|document|file)',
            r'according to the (\w+) (?:invoice|document|file)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, answer_lower, re.IGNORECASE)
            for match in matches:
                # Normalize the match
                doc_name = match if match.endswith('.pdf') else f"{match}.pdf"
                
                # Check if this document is in valid sources
                if doc_name not in {s.lower() for s in valid_sources}:
                    invalid.append(doc_name)
        
        return list(set(invalid))  # Remove duplicates
    
    def _create_citations(self, sources: List[Dict[str, Any]]) -> str:
        """
        Create formatted citations.
        
        Args:
            sources: List of source documents
            
        Returns:
            Formatted citation string
        """
        if not sources:
            return ""
        
        citations = ["Sources:"]
        for i, src in enumerate(sources[:3], 1):  # Top 3 sources
            source_file = src['metadata'].get('source_file', 'unknown')
            page = src['metadata'].get('page', 'unknown')
            score = src.get('similarity_score', 0)
            citations.append(f"[{i}] {source_file} (page {page}, relevance: {score:.2f})")
        
        return "\n".join(citations)
    
    def _format_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format sources for response.
        
        Args:
            sources: Raw source documents
            
        Returns:
            Formatted sources
        """
        formatted = []
        for src in sources:
            formatted.append({
                'source': src['metadata'].get('source_file', 'unknown'),
                'page': src['metadata'].get('page', 'unknown'),
                'score': src.get('similarity_score', 0),
                'preview': src['content'][:200] + '...' if len(src['content']) > 200 else src['content'],
                'content': src['content']  # Include full content
            })
        return formatted
    
    def _add_to_history(self, question: str, response: Dict[str, Any]):
        """
        Add query to history.
        
        Args:
            question: User question
            response: System response
        """
        self.history.append({
            'timestamp': time.time(),
            'question': question,
            'answer': response['answer'],
            'confidence': response['confidence'],
            'retrieved_docs': response['retrieved_docs']
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_queries': len(self.history),
            'cache_size': len(self.cache) if self.cache else 0,
            'vector_store_docs': self.vector_store.collection.count(),
            'avg_confidence': sum(h['confidence'] for h in self.history) / len(self.history) if self.history else 0,
            'embedding_model': self.config.embedding.model_name,
            'llm_model': self.config.llm.model_name
        }
    
    def clear_cache(self):
        """Clear query cache."""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Cache cleared")