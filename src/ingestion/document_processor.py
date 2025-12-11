"""
Document processing module for loading and chunking documents.
"""
from typing import List, Optional
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.rag.config import ChunkingConfig
from src.rag.utils import get_logger, clean_text, compute_text_hash, calculate_text_stats, has_tables


logger = get_logger(__name__)


class DocumentLoader:
    """Handles loading documents from various sources."""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def load_pdf(self, file_path: Path) -> List[Document]:
        """
        Load a single PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects
        """
        try:
            logger.info(f"Loading PDF: {file_path.name}")
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # Enhance metadata
            for i, doc in enumerate(documents):
                doc.metadata['source_file'] = file_path.name
                doc.metadata['file_type'] = 'pdf'
                doc.metadata['file_path'] = str(file_path)
                doc.metadata['page_number'] = i
            
            logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")
            return []
    
    def load_directory(self, directory: Path, recursive: bool = True) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to search recursively
            
        Returns:
            List of all loaded documents
        """
        all_documents = []
        
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(directory.glob(pattern))
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            documents = self.load_pdf(pdf_file)
            all_documents.extend(documents)
        
        logger.info(f"Total pages loaded: {len(all_documents)}")
        return all_documents


class DocumentChunker:
    """Handles intelligent document chunking."""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=config.separators
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks with enhanced metadata.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        logger.info(f"Chunking {len(documents)} documents...")
        
        # Adaptive chunking based on content
        all_chunks = []
        
        for doc in documents:
            # Check if document has tables
            if has_tables(doc.page_content):
                # Use larger chunks for tables
                table_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.chunk_size * 2,
                    chunk_overlap=self.config.chunk_overlap,
                    separators=["\n\n", "\n"]
                )
                chunks = table_splitter.split_documents([doc])
            else:
                chunks = self.text_splitter.split_documents([doc])
            
            # Enhance chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
                chunk.metadata['total_chunks'] = len(chunks)
                chunk.metadata['parent_page'] = doc.metadata.get('page', 0)
                
                # Add text statistics
                stats = calculate_text_stats(chunk.page_content)
                chunk.metadata.update(stats)
                
                # Add content hash for deduplication
                chunk.metadata['content_hash'] = compute_text_hash(chunk.page_content)
            
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        if all_chunks:
            logger.info(f"Example chunk preview: {all_chunks[0].page_content[:150]}...")
        
        return all_chunks


class DocumentProcessor:
    """Main document processing pipeline."""
    
    def __init__(self, chunking_config: ChunkingConfig):
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker(chunking_config)
    
    def process_directory(
        self, 
        directory: Path, 
        deduplicate: bool = True,
        clean: bool = True
    ) -> List[Document]:
        """
        Complete processing pipeline for a directory.
        
        Args:
            directory: Directory containing documents
            deduplicate: Whether to remove duplicate chunks
            clean: Whether to clean text
            
        Returns:
            List of processed document chunks
        """
        # Load documents
        documents = self.loader.load_directory(directory)
        
        if not documents:
            logger.warning("No documents loaded")
            return []
        
        # Clean text if requested
        if clean:
            for doc in documents:
                doc.page_content = clean_text(doc.page_content)
        
        # Chunk documents
        chunks = self.chunker.chunk_documents(documents)
        
        # Deduplicate if requested
        if deduplicate:
            chunks = self._deduplicate_chunks(chunks)
        
        logger.info(f"Final chunk count: {len(chunks)}")
        return chunks
    
    def process_file(self, file_path: Path) -> List[Document]:
        """
        Process a single file.
        
        Args:
            file_path: Path to file
            
        Returns:
            List of processed chunks
        """
        documents = self.loader.load_pdf(file_path)
        
        if not documents:
            return []
        
        chunks = self.chunker.chunk_documents(documents)
        return chunks
    
    def _deduplicate_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        Remove duplicate chunks based on content hash.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Deduplicated chunks
        """
        seen_hashes = set()
        unique_chunks = []
        duplicates = 0
        
        for chunk in chunks:
            content_hash = chunk.metadata.get('content_hash')
            if content_hash and content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_chunks.append(chunk)
            else:
                duplicates += 1
        
        if duplicates > 0:
            logger.info(f"Removed {duplicates} duplicate chunks")
        
        return unique_chunks