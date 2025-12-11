"""
Evaluation and monitoring module for RAG pipeline.
"""
import time
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from src.rag.utils import get_logger


logger = get_logger(__name__)


class RAGEvaluator:
    """Evaluates RAG pipeline performance and quality."""
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("./data/evaluation_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history = []
        self.current_session = {
            'start_time': time.time(),
            'queries': []
        }
    
    def evaluate_response(
        self,
        query: str,
        answer: str,
        context: str,
        sources: List[Dict[str, Any]],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single RAG response.
        
        Args:
            query: User query
            answer: Generated answer
            context: Retrieved context
            sources: Source documents
            ground_truth: Optional ground truth answer
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # 1. Faithfulness: Is answer grounded in context?
        metrics['faithfulness'] = self._check_faithfulness(answer, context)
        
        # 2. Relevance: Does answer address the query?
        metrics['relevance'] = self._check_relevance(query, answer)
        
        # 3. Completeness: Does context contain answer?
        metrics['completeness'] = self._check_completeness(query, context)
        
        # 4. Source quality
        metrics['source_quality'] = self._evaluate_sources(sources)
        
        # 5. Answer length appropriateness
        metrics['length_score'] = self._evaluate_length(answer)
        
        # 6. If ground truth available, compute accuracy
        if ground_truth:
            metrics['accuracy'] = self._compute_accuracy(answer, ground_truth)
        
        # Overall score
        metrics['overall_score'] = sum(metrics.values()) / len(metrics)
        
        # Log metrics
        self._log_evaluation(query, answer, metrics)
        
        return metrics
    
    def _check_faithfulness(self, answer: str, context: str) -> float:
        """
        Check if answer is grounded in context.
        
        Args:
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Faithfulness score (0-1)
        """
        if not answer or not context:
            return 0.0
        
        # Check for hallucination indicators
        hallucination_phrases = [
            'i cannot answer',
            'not mentioned',
            'unclear',
            'not enough information',
            'based on the available documents'
        ]
        
        answer_lower = answer.lower()
        
        # If answer explicitly states uncertainty, high faithfulness
        if any(phrase in answer_lower for phrase in hallucination_phrases):
            return 1.0
        
        # Check if answer phrases exist in context
        answer_sentences = [s.strip() for s in answer.split('.') if s.strip()]
        context_lower = context.lower()
        
        grounded_sentences = 0
        for sentence in answer_sentences:
            # Check if key words from sentence appear in context
            words = set(sentence.lower().split())
            # Remove common words
            words = words - {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for'}
            
            if words:
                overlap = sum(1 for word in words if word in context_lower)
                if overlap / len(words) > 0.5:  # 50% of words found
                    grounded_sentences += 1
        
        faithfulness = grounded_sentences / len(answer_sentences) if answer_sentences else 0.0
        return faithfulness
    
    def _check_relevance(self, query: str, answer: str) -> float:
        """
        Check if answer is relevant to query.
        
        Args:
            query: User query
            answer: Generated answer
            
        Returns:
            Relevance score (0-1)
        """
        if not query or not answer:
            return 0.0
        
        # Extract key terms from query
        query_words = set(query.lower().split())
        query_words = query_words - {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'the', 'a'}
        
        answer_lower = answer.lower()
        
        # Check how many query terms appear in answer
        matching_terms = sum(1 for word in query_words if word in answer_lower)
        
        relevance = matching_terms / len(query_words) if query_words else 0.0
        return min(relevance, 1.0)
    
    def _check_completeness(self, query: str, context: str) -> float:
        """
        Check if context likely contains answer to query.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Completeness score (0-1)
        """
        if not query or not context:
            return 0.0
        
        # Extract key terms from query
        query_words = set(query.lower().split())
        query_words = query_words - {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'the', 'a'}
        
        context_lower = context.lower()
        
        # Check how many query terms appear in context
        matching_terms = sum(1 for word in query_words if word in context_lower)
        
        completeness = matching_terms / len(query_words) if query_words else 0.0
        return min(completeness, 1.0)
    
    def _evaluate_sources(self, sources: List[Dict[str, Any]]) -> float:
        """
        Evaluate quality of retrieved sources.
        
        Args:
            sources: List of source documents
            
        Returns:
            Source quality score (0-1)
        """
        if not sources:
            return 0.0
        
        # Check average similarity score
        avg_score = sum(src.get('score', 0) for src in sources) / len(sources)
        
        # Penalize if too few sources
        count_penalty = min(len(sources) / 3, 1.0)
        
        return avg_score * count_penalty
    
    def _evaluate_length(self, answer: str) -> float:
        """
        Evaluate if answer length is appropriate.
        
        Args:
            answer: Generated answer
            
        Returns:
            Length score (0-1)
        """
        word_count = len(answer.split())
        
        # Ideal range: 20-200 words
        if 20 <= word_count <= 200:
            return 1.0
        elif word_count < 20:
            return word_count / 20
        else:
            return max(0.5, 1.0 - (word_count - 200) / 300)
    
    def _compute_accuracy(self, answer: str, ground_truth: str) -> float:
        """
        Compute accuracy against ground truth.
        
        Args:
            answer: Generated answer
            ground_truth: Ground truth answer
            
        Returns:
            Accuracy score (0-1)
        """
        # Simple word overlap metric
        answer_words = set(answer.lower().split())
        truth_words = set(ground_truth.lower().split())
        
        if not truth_words:
            return 0.0
        
        overlap = len(answer_words & truth_words)
        accuracy = overlap / len(truth_words)
        
        return min(accuracy, 1.0)
    
    def _log_evaluation(self, query: str, answer: str, metrics: Dict[str, float]):
        """
        Log evaluation metrics.
        
        Args:
            query: User query
            answer: Generated answer
            metrics: Evaluation metrics
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'answer': answer[:200] + '...' if len(answer) > 200 else answer,
            'metrics': metrics
        }
        
        self.metrics_history.append(log_entry)
        self.current_session['queries'].append(log_entry)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics for current session.
        
        Returns:
            Dictionary with session statistics
        """
        if not self.current_session['queries']:
            return {'message': 'No queries in current session'}
        
        queries = self.current_session['queries']
        
        # Aggregate metrics
        all_metrics = [q['metrics'] for q in queries]
        
        stats = {
            'session_duration': time.time() - self.current_session['start_time'],
            'total_queries': len(queries),
            'avg_faithfulness': sum(m.get('faithfulness', 0) for m in all_metrics) / len(all_metrics),
            'avg_relevance': sum(m.get('relevance', 0) for m in all_metrics) / len(all_metrics),
            'avg_completeness': sum(m.get('completeness', 0) for m in all_metrics) / len(all_metrics),
            'avg_overall_score': sum(m.get('overall_score', 0) for m in all_metrics) / len(all_metrics)
        }
        
        return stats
    
    def save_session_log(self):
        """Save current session to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f"session_{timestamp}.json"
        
        session_data = {
            'session_info': {
                'start_time': self.current_session['start_time'],
                'end_time': time.time(),
                'duration': time.time() - self.current_session['start_time']
            },
            'queries': self.current_session['queries'],
            'stats': self.get_session_stats()
        }
        
        with open(log_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Session log saved to {log_file}")
    
    def get_all_time_stats(self) -> Dict[str, Any]:
        """
        Get all-time statistics.
        
        Returns:
            Dictionary with all-time statistics
        """
        if not self.metrics_history:
            return {'message': 'No evaluation history'}
        
        all_metrics = [entry['metrics'] for entry in self.metrics_history]
        
        return {
            'total_evaluations': len(self.metrics_history),
            'avg_faithfulness': sum(m.get('faithfulness', 0) for m in all_metrics) / len(all_metrics),
            'avg_relevance': sum(m.get('relevance', 0) for m in all_metrics) / len(all_metrics),
            'avg_completeness': sum(m.get('completeness', 0) for m in all_metrics) / len(all_metrics),
            'avg_overall_score': sum(m.get('overall_score', 0) for m in all_metrics) / len(all_metrics)
        }