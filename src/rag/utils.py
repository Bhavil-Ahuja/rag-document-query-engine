"""
Utility functions for the RAG pipeline.
"""
import hashlib
import re
from typing import List, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\[\]\:\;]', '', text)
    
    return text.strip()


def compute_text_hash(text: str) -> str:
    """
    Compute MD5 hash of text for deduplication.
    
    Args:
        text: Text string
        
    Returns:
        MD5 hash hex string
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def compute_query_hash(query: str, params: dict) -> str:
    """
    Compute cache key for query.
    
    Args:
        query: Query string
        params: Query parameters
        
    Returns:
        Cache key
    """
    key_str = f"{query}_{str(sorted(params.items()))}"
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_sources(sources: List[dict]) -> str:
    """
    Format source citations.
    
    Args:
        sources: List of source dictionaries
        
    Returns:
        Formatted citation string
    """
    if not sources:
        return ""
    
    citations = []
    for i, src in enumerate(sources, 1):
        source_file = src.get('source', 'unknown')
        page = src.get('page', 'unknown')
        score = src.get('score', 0.0)
        citations.append(f"[{i}] {source_file} (page {page}, relevance: {score:.2f})")
    
    return "\n".join(citations)


def has_tables(text: str) -> bool:
    """
    Check if text contains table-like structures.
    
    Args:
        text: Text to check
        
    Returns:
        True if tables detected
    """
    # Simple heuristics
    pipe_count = text.count('|')
    tab_count = text.count('\t')
    
    return pipe_count > 5 or tab_count > 3


def extract_numbers(text: str) -> List[str]:
    """
    Extract numbers from text.
    
    Args:
        text: Text to extract from
        
    Returns:
        List of number strings
    """
    return re.findall(r'\d+\.?\d*', text)


def calculate_text_stats(text: str) -> dict:
    """
    Calculate statistics about text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of statistics
    """
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    return {
        'char_count': len(text),
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
        'has_numbers': bool(re.search(r'\d', text)),
        'has_tables': has_tables(text)
    }