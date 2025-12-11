"""
Query optimization and routing module.
"""
from typing import List, Dict, Any
from langchain_groq import ChatGroq

from src.rag.config import LLMConfig
from src.rag.utils import get_logger


logger = get_logger(__name__)


class QueryOptimizer:
    """Optimizes and transforms queries for better retrieval."""
    
    def __init__(self, llm_config: LLMConfig):
        self.llm = ChatGroq(
            groq_api_key=llm_config.api_key,
            model_name=llm_config.model_name,
            temperature=0.1,
            max_tokens=512
        )
        self.llm_creative = ChatGroq(
            groq_api_key=llm_config.api_key,
            model_name=llm_config.model_name,
            temperature=0.7,  # Higher temperature for creative generation
            max_tokens=1024
        )
    
    def extract_document_reference(self, query: str) -> Dict[str, Any]:
        """
        Extract document reference from query for filtering.
        
        Examples:
        - "What is total in Lenskart invoice?" → {"keywords": ["lenskart"]}
        - "Show me Feb electricity bill" → {"keywords": ["feb", "electricity"]}
        - "What is the total amount?" → {"keywords": []}
        
        Args:
            query: User query
            
        Returns:
            Dictionary with document filtering hints
        """
        import re
        
        query_lower = query.lower()
        
        # Common document identifiers
        hints = {
            'keywords': [],
            'months': [],
            'has_specific_doc': False
        }
        
        # Extract potential document names/keywords
        # Look for capitalized words or quoted text
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', query)
        hints['keywords'].extend([w.lower() for w in capitalized])
        
        # Extract months
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                  'july', 'august', 'september', 'october', 'november', 'december',
                  'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        
        for month in months:
            if month in query_lower:
                hints['months'].append(month)
                hints['has_specific_doc'] = True
        
        # Look for document type indicators
        doc_types = ['invoice', 'bill', 'receipt', 'statement', 'order', 'purchase']
        for doc_type in doc_types:
            if doc_type in query_lower:
                hints['keywords'].append(doc_type)
        
        # Check if query mentions specific document
        if hints['keywords'] or hints['months']:
            hints['has_specific_doc'] = True
        
        return hints
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Break complex queries into simpler sub-queries.
        
        Args:
            query: Original complex query
            
        Returns:
            List of sub-queries
        """
        logger.info(f"Decomposing query: {query}")
        
        prompt = f"""Break this complex question into 2-3 simpler, specific sub-questions that together would answer the original question.
Return ONLY the sub-questions, one per line, without numbering or extra text.

Original question: {query}

Sub-questions:"""
        
        try:
            response = self.llm.invoke([prompt])
            sub_queries = [
                q.strip() 
                for q in response.content.split('\n') 
                if q.strip() and not q.strip().startswith(('1.', '2.', '3.', '-', '*'))
            ]
            
            # Clean up any remaining numbering
            sub_queries = [q.lstrip('123456789.-*) ') for q in sub_queries]
            
            if sub_queries:
                logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
                return sub_queries
            else:
                logger.warning("Decomposition failed, returning original query")
                return [query]
                
        except Exception as e:
            logger.error(f"Error decomposing query: {e}")
            return [query]
    
    def expand_query(self, query: str) -> List[str]:
        """
        Generate alternative phrasings of the query.
        
        Args:
            query: Original query
            
        Returns:
            List of query variations (including original)
        """
        logger.info(f"Expanding query: {query}")
        
        prompt = f"""Generate 2 alternative ways to ask this question. Use different words but keep the same meaning.
Return ONLY the alternative questions, one per line, without numbering or extra text.

Original: {query}

Alternatives:"""
        
        try:
            response = self.llm.invoke([prompt])
            alternatives = [
                q.strip() 
                for q in response.content.split('\n') 
                if q.strip() and q.strip() != query
            ]
            
            # Clean up numbering
            alternatives = [q.lstrip('123456789.-*) ') for q in alternatives]
            
            # Include original query
            all_queries = [query] + alternatives
            
            logger.info(f"Expanded to {len(all_queries)} query variations")
            return all_queries
            
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return [query]
    
    def generate_hyde_document(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query (Hyde).
        
        This technique generates a hypothetical answer/document, then searches
        for real documents similar to it. Often more effective than searching
        with the query directly, especially for complex or conceptual questions.
        
        Args:
            query: User query
            
        Returns:
            Hypothetical document text
        """
        logger.info(f"Generating Hyde document for: {query}")
        
        prompt = f"""You are an expert writer. Generate a detailed, well-structured paragraph that would perfectly answer this question.
Write as if you're creating a section of a document that contains the answer.
Be specific, use technical terms, and include relevant details.

Question: {query}

Generated document paragraph:"""
        
        try:
            response = self.llm_creative.invoke([prompt])
            hyde_doc = response.content.strip()
            
            logger.info(f"Generated Hyde document ({len(hyde_doc)} chars)")
            return hyde_doc
            
        except Exception as e:
            logger.error(f"Error generating Hyde document: {e}")
            return query
    
    def generate_stepback_question(self, query: str) -> str:
        """
        Generate a broader "step-back" question for better context retrieval.
        
        StepBack prompting asks a higher-level, more general question before
        the specific one. This helps retrieve broader context that aids in
        answering the specific question.
        
        Args:
            query: Original specific query
            
        Returns:
            Broader step-back question
        """
        logger.info(f"Generating step-back question for: {query}")
        
        prompt = f"""Given a specific question, generate a broader, more general question that would help provide context for answering it.
The step-back question should ask about underlying concepts, principles, or categories rather than specific details.

Specific question: {query}

Broader step-back question:"""
        
        try:
            response = self.llm.invoke([prompt])
            stepback = response.content.strip()
            
            # Clean up any extra text
            stepback = stepback.replace("Broader question:", "").replace("Step-back question:", "").strip()
            
            logger.info(f"Step-back question: {stepback}")
            return stepback
            
        except Exception as e:
            logger.error(f"Error generating step-back question: {e}")
            return query
    
    def generate_query2doc(self, query: str) -> str:
        """
        Generate a pseudo-document to augment the query (Query2Doc).
        
        Similar to Hyde but focuses on generating a full pseudo-document
        with context, background, and potential answer content that can
        be used to augment the original query.
        
        Args:
            query: User query
            
        Returns:
            Generated pseudo-document
        """
        logger.info(f"Generating Query2Doc for: {query}")
        
        prompt = f"""Generate a comprehensive document section that would contain information relevant to answering this question.
Include background context, related concepts, and the type of information that would appear in a document containing the answer.

Question: {query}

Generated document section:"""
        
        try:
            response = self.llm_creative.invoke([prompt])
            pseudo_doc = response.content.strip()
            
            logger.info(f"Generated Query2Doc ({len(pseudo_doc)} chars)")
            return pseudo_doc
            
        except Exception as e:
            logger.error(f"Error generating Query2Doc: {e}")
            return query
    
    def classify_query(self, query: str) -> str:
        """
        Classify the query type for routing.
        
        Args:
            query: Query string
            
        Returns:
            Query type (factual, comparison, summary, temporal, analytical)
        """
        query_lower = query.lower()
        
        # Simple rule-based classification
        if any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs', 'better', 'worse']):
            return 'comparison'
        elif any(word in query_lower for word in ['summarize', 'summary', 'overview', 'about', 'what is']):
            return 'summary'
        elif any(word in query_lower for word in ['when', 'date', 'time', 'year', 'month', 'day', 'purchase', 'bought', 'ordered']):
            return 'temporal'
        elif any(word in query_lower for word in ['why', 'how', 'explain', 'reason']):
            return 'analytical'
        elif any(word in query_lower for word in ['list', 'enumerate', 'all', 'every']):
            return 'list'
        else:
            return 'factual'
    
    
    def optimize_query_batch(
        self, 
        query: str,
        use_hyde: bool = False,
        use_stepback: bool = False,
        use_query2doc: bool = False,
        use_multiquery: bool = True
    ) -> Dict[str, Any]:
        """
        Perform complete query optimization with ONE combined LLM call.
        
        This method generates all optimization outputs in a single LLM call,
        then filters based on flags. Much more efficient than separate calls.
        
        Args:
            query: Original query
            use_hyde: Use Hyde (Hypothetical Document Embeddings)
            use_stepback: Use Step-back prompting
            use_query2doc: Use Query2Doc augmentation
            use_multiquery: Use MultiQuery variations
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing query with batch generation: {query}")
        
        # Classify query
        query_type = self.classify_query(query)
        logger.info(f"Query type: {query_type}")
        
        # Build combined prompt that generates everything at once
        # Add query-type specific guidance
        query_guidance = ""
        if query_type == 'temporal':
            query_guidance = "\nNote: This is a TEMPORAL query asking about dates/time. Generate variations that explicitly include date-related terms and formats that might appear in documents."
        elif query_type == 'factual':
            query_guidance = "\nNote: This is a FACTUAL query. Generate variations using different synonyms and phrasings."
        elif query_type == 'analytical':
            query_guidance = "\nNote: This is an ANALYTICAL query requiring explanation. Generate conceptual variations."
        
        prompt = f"""Given the following user question, generate multiple optimized retrieval queries and documents.

Original Question: {query}
Query Type: {query_type}{query_guidance}

Please provide the following in this EXACT format:

1. QUERY VARIATIONS (2-3 SHORT alternative phrasings - just the question, nothing else):
What is the [relevant noun]?
When did [action] happen?
[Another short variation]

2. HYPOTHETICAL DOCUMENT (A realistic paragraph that would contain the answer - mimic actual document style):
[Write a realistic paragraph here]

3. STEP-BACK QUESTION (A broader, more general question - just the question):
[Write broader question here]

4. PSEUDO DOCUMENT (Background context):
[Write background info here]

CRITICAL: For QUERY VARIATIONS, output ONLY the alternative questions, one per line. Do NOT add explanations or descriptions.

Example for "When did I purchase from Decathlon?":
1. QUERY VARIATIONS:
What is the Decathlon purchase date?
Decathlon order date
When was the Decathlon transaction?

Now generate for the actual query above."""

        try:
            # Single LLM call for everything
            response = self.llm_creative.invoke([prompt])
            content = response.content.strip()
            
            # Parse the response
            parsed = self._parse_batch_response(content)
            
            # Initialize optimization result
            optimization = {
                'original': query,
                'type': query_type,
                'variations': [query],
                'sub_queries': None,
                'hyde_document': None,
                'stepback_question': None,
                'query2doc': None,
                'strategy': 'single'
            }
            
            # Use only the parts we need based on flags
            if use_multiquery and parsed.get('variations'):
                optimization['variations'] = [query] + parsed['variations']
                optimization['strategy'] = 'multiquery'
                logger.info(f"Using {len(optimization['variations'])} query variations")
            
            # No special handling needed - batch optimization handles all query types
            # The LLM is smart enough to understand temporal, factual, analytical queries
            # without hardcoded domain-specific logic
            
            if use_hyde and parsed.get('hyde_document'):
                optimization['hyde_document'] = parsed['hyde_document']
                if optimization['strategy'] != 'single':
                    optimization['strategy'] += '+hyde'
                else:
                    optimization['strategy'] = 'hyde'
                logger.info("Using Hyde document")
            
            if use_stepback and parsed.get('stepback_question'):
                optimization['stepback_question'] = parsed['stepback_question']
                if optimization['strategy'] != 'single':
                    optimization['strategy'] += '+stepback'
                else:
                    optimization['strategy'] = 'stepback'
                logger.info("Using StepBack question")
            
            if use_query2doc and parsed.get('query2doc'):
                optimization['query2doc'] = parsed['query2doc']
                logger.info("Using Query2Doc")
            
            logger.info(f"Optimization strategy: {optimization['strategy']}")
            
            return optimization
            
        except Exception as e:
            logger.error(f"Error in batch optimization: {e}")
            # Fallback to simple optimization
            return {
                'original': query,
                'type': query_type,
                'variations': [query],
                'sub_queries': None,
                'hyde_document': None,
                'stepback_question': None,
                'query2doc': None,
                'strategy': 'single'
            }
    
    def _parse_batch_response(self, content: str) -> Dict[str, Any]:
        """
        Parse the batch optimization response.
        
        Args:
            content: LLM response containing all optimization outputs
            
        Returns:
            Dictionary with parsed components
        """
        result = {
            'variations': [],
            'hyde_document': None,
            'stepback_question': None,
            'query2doc': None
        }
        
        # Split by sections
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect section headers
            if 'query variation' in line_lower or 'variation' in line_lower:
                if current_section and current_content:
                    result = self._save_section(result, current_section, current_content)
                current_section = 'variations'
                current_content = []
            elif 'hypothetical document' in line_lower or 'hyde' in line_lower:
                if current_section and current_content:
                    result = self._save_section(result, current_section, current_content)
                current_section = 'hyde'
                current_content = []
            elif 'step-back' in line_lower or 'stepback' in line_lower:
                if current_section and current_content:
                    result = self._save_section(result, current_section, current_content)
                current_section = 'stepback'
                current_content = []
            elif 'pseudo document' in line_lower or 'query2doc' in line_lower:
                if current_section and current_content:
                    result = self._save_section(result, current_section, current_content)
                current_section = 'query2doc'
                current_content = []
            # Skip markers and empty lines
            elif line.strip() and not line.startswith('[') and not line.startswith('#'):
                current_content.append(line.strip())
        
        # Save last section
        if current_section and current_content:
            result = self._save_section(result, current_section, current_content)
        
        return result
    
    def _save_section(self, result: dict, section: str, content: list) -> dict:
        """Save parsed section content."""
        text = ' '.join(content).strip()
        
        if section == 'variations':
            # Extract individual variations
            variations = [v.strip() for v in text.split('\n') if v.strip()]
            result['variations'].extend(variations[:2])  # Max 2 variations
        elif section == 'hyde':
            result['hyde_document'] = text
        elif section == 'stepback':
            result['stepback_question'] = text
        elif section == 'query2doc':
            result['query2doc'] = text
        
        return result
    
    def optimize_query(
        self, 
        query: str,
        use_hyde: bool = False,
        use_stepback: bool = False,
        use_query2doc: bool = False,
        use_multiquery: bool = True
    ) -> Dict[str, Any]:
        """
        Perform complete query optimization (delegates to batch method).
        
        This is now a wrapper that calls optimize_query_batch for efficiency.
        ONE LLM call instead of multiple!
        
        Args:
            query: Original query
            use_hyde: Use Hyde (Hypothetical Document Embeddings)
            use_stepback: Use Step-back prompting
            use_query2doc: Use Query2Doc augmentation
            use_multiquery: Use MultiQuery variations
            
        Returns:
            Dictionary with optimization results
        """
        return self.optimize_query_batch(
            query=query,
            use_hyde=use_hyde,
            use_stepback=use_stepback,
            use_query2doc=use_query2doc,
            use_multiquery=use_multiquery
        )


class QueryRouter:
    """Routes queries to appropriate retrieval strategies."""
    
    def __init__(self):
        self.route_map = {
            'factual': {'top_k': 5, 'rerank': True},
            'comparison': {'top_k': 10, 'rerank': True},
            'summary': {'top_k': 8, 'rerank': False},
            'temporal': {'top_k': 15, 'rerank': False},  # DISABLED re-ranker for temporal - dates are in form fields!
            'analytical': {'top_k': 10, 'rerank': True},
            'list': {'top_k': 15, 'rerank': False}
        }
    
    def route(self, query_type: str) -> Dict[str, Any]:
        """
        Get retrieval parameters based on query type.
        
        Args:
            query_type: Type of query
            
        Returns:
            Dictionary with retrieval parameters
        """
        params = self.route_map.get(query_type, {'top_k': 5, 'rerank': True})
        logger.info(f"Routing {query_type} query with params: {params}")
        return params