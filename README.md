# Production-Grade RAG Pipeline 🚀

A modular, production-ready Retrieval-Augmented Generation (RAG) system with advanced features including hybrid search, re-ranking, hallucination prevention, and comprehensive evaluation.

## ✨ Features

### Core Capabilities
- 📄 **Multi-format Document Processing**: PDF support with intelligent chunking
- 🔍 **Hybrid Search**: Combines semantic (vector) and keyword (BM25) search
- 🎯 **Re-ranking**: Cross-encoder re-ranking for improved relevance
- 🛡️ **Hallucination Prevention**: Strict grounding with confidence thresholds
- 📊 **Evaluation & Monitoring**: Built-in metrics for faithfulness, relevance, and completeness
- 💾 **Persistent Storage**: ChromaDB vector database with caching
- ⚡ **Query Optimization**: Automatic query decomposition and expansion

### Advanced Features
- **Adaptive Chunking**: Context-aware text splitting (handles tables differently)
- **Deduplication**: Automatic removal of duplicate content
- **Source Citations**: Automatic citation generation with confidence scores
- **Query Routing**: Type-based retrieval parameter optimization
- **Session Logging**: Comprehensive evaluation logs
- **Metadata Enrichment**: Rich metadata tracking for better retrieval

## 📁 Project Structure

```
RAG/
├── config.py                 # Configuration management
├── utils.py                  # Utility functions
├── document_processor.py     # Document loading & chunking
├── embedding_manager.py      # Embedding generation
├── vector_store.py          # Vector database management
├── retriever.py             # Hybrid retrieval & re-ranking
├── query_optimizer.py       # Query optimization & routing
├── rag_pipeline.py          # Main RAG pipeline
├── evaluator.py             # Evaluation & monitoring
├── main.py                  # Entry point & orchestration
├── requirements.txt         # Dependencies
└── data/
    ├── pdf_files/          # Input PDFs
    ├── vector_store/       # ChromaDB storage
    ├── cache/              # Query cache
    └── evaluation_logs/    # Evaluation logs
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Basic Usage

```python
from pathlib import Path
from main import RAGSystem

# Initialize system
rag_system = RAGSystem()

# Ingest documents
pdf_dir = Path("./data/pdf_files")
rag_system.ingest_documents_from_directory(pdf_dir)

# Query
response = rag_system.query("What is this document about?")
print(response['answer'])
print(f"Confidence: {response['confidence']:.2f}")
```

### 4. Run the System

**Interactive Mode (Recommended):**
```bash
python main.py
# or
python run_interactive.py
```

**Batch Mode (for testing):**
```bash
python run_batch.py
```

**Web UI (Streamlit):**
```bash
# Basic app
streamlit run app.py

# Advanced app (with collections & PDF viewer)
streamlit run app_advanced.py
```

## 🎨 Web UI (Streamlit)

### Basic App (`app.py`)

Beautiful web interface for document upload and chat:

**Features:**
- 📤 **Drag & Drop Upload**: Upload multiple PDFs at once
- 💬 **Interactive Chat**: Natural conversation interface
- 📊 **Rich Metadata**: View confidence, sources, and quality scores
- 📈 **Real-time Stats**: Monitor system performance
- 🎯 **Source Citations**: See which documents were used
- 💾 **Session Persistence**: Chat history maintained during session

**Quick Start:**
```bash
# Install streamlit (if not already installed)
pip install streamlit

# Run the web app
streamlit run app.py

# App opens at http://localhost:8501
```

**Usage:**
1. Upload PDFs using the sidebar
2. Click "Process" to ingest documents
3. Ask questions in the chat interface
4. View answers with sources and confidence scores
5. Click "View Details" for quality metrics

For detailed guide, see [STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md)

### Advanced App (`app_advanced.py`)

**Enhanced features for power users:**

- 📁 **Multiple Collections**: Organize documents into folders/collections
- 📄 **PDF Viewer**: View documents directly in browser
- 🎯 **Collection-Specific Queries**: Query within specific collections only
- 📋 **Document Management**: See all uploaded files, view PDFs
- 📊 **Advanced Statistics**: Per-collection and system-wide metrics

**Quick Start:**
```bash
streamlit run app_advanced.py
```

**Use Cases:**
- Multi-department organizations (finance, legal, HR)
- Time-based organization (2023 docs, 2024 docs)
- Topic-based collections (invoices, contracts, reports)
- Large document sets requiring organization

**Example:**
```
Collections:
├── invoices          # Upload all invoices here
├── contracts         # Upload all contracts here  
└── reports           # Upload all reports here

Query "invoices" collection → Get answers from invoices only!
```

For detailed guide, see [ADVANCED_APP_GUIDE.md](ADVANCED_APP_GUIDE.md)

## 💬 Interactive Mode (Terminal)

The interactive terminal provides a chat-like interface:

**Available Commands:**
- Type any question and press Enter
- `stats` - Show system statistics
- `history` - Show last 10 queries
- `clear` - Clear query cache
- `help` - Show help message
- `quit` or `exit` - Exit gracefully

**Example Session:**
```
💬 You: What is the Lenskart invoice about?

🔍 Processing query 1...

🤖 Assistant:
----------------------------------------------------------------------
The Lenskart pdf is a tax invoice for shipment code 
SNXS2260000035818963, order #1322894003, dated 03/11/2025, 
with a total quantity of 3 items. Payment method is COD.

Sources:
[1] Lenskart 1.pdf (page 0, relevance: 89.2%)
----------------------------------------------------------------------

📈 Metadata:
   Confidence: 89%
   Retrieved docs: 3
   Query time: 2.34s

📊 Quality Scores:
   Faithfulness: 95%
   Relevance: 88%
   Overall: 91%

💬 You: stats

📊 System Statistics:
----------------------------------------------------------------------
Pipeline:
  Total queries: 1
  Average confidence: 89%
  Cache size: 1 queries
  Documents in vector store: 14

Models:
  Embedding: BAAI/bge-large-en-v1.5
  LLM: llama-3.1-8b-instant
...

💬 You: quit

👋 Saving evaluation log and exiting...
```

## 🔧 Configuration

Customize the RAG pipeline in `config.py`:

```python
from config import RAGConfig, EmbeddingConfig, RetrievalConfig

config = RAGConfig(
    embedding=EmbeddingConfig(
        model_name="BAAI/bge-large-en-v1.5",  # Better embeddings
        dimension=1024
    ),
    retrieval=RetrievalConfig(
        top_k=10,
        min_similarity_score=0.3,
        use_reranker=True,
        reranker_top_k=5
    ),
    min_confidence_threshold=0.5,  # Hallucination prevention
    enable_caching=True,
    enable_evaluation=True
)

rag_system = RAGSystem(config)
```

## 📚 Advanced Usage

### Custom Document Processing

```python
from document_processor import DocumentProcessor
from config import ChunkingConfig

# Custom chunking
chunking_config = ChunkingConfig(
    chunk_size=1500,
    chunk_overlap=300
)

processor = DocumentProcessor(chunking_config)
chunks = processor.process_directory(
    Path("./data/pdf_files"),
    deduplicate=True,
    clean=True
)
```

### Query with Evaluation

```python
response = rag_system.query(
    "What products are mentioned?",
    top_k=10,
    min_score=0.4,
    evaluate=True
)

print(f"Faithfulness: {response['evaluation']['faithfulness']:.2f}")
print(f"Relevance: {response['evaluation']['relevance']:.2f}")
```

### Access Statistics

```python
stats = rag_system.get_stats()
print(f"Total queries: {stats['pipeline']['total_queries']}")
print(f"Avg confidence: {stats['pipeline']['avg_confidence']:.2f}")
print(f"Vector store docs: {stats['vector_store']['count']}")
```

## 🎯 Key Components

### 1. Document Processor
- Loads PDFs with metadata enrichment
- Intelligent chunking (adaptive for tables)
- Text cleaning and deduplication

### 2. Embedding Manager
- Supports multiple models (MiniLM, BGE, OpenAI)
- Batch processing with progress tracking
- Similarity computation utilities

### 3. Vector Store
- ChromaDB persistent storage
- Batch operations for efficiency
- Metadata filtering support

### 4. Hybrid Retriever
- Semantic search (vector similarity)
- Keyword search (BM25)
- Cross-encoder re-ranking
- Configurable hybrid weighting

### 5. Query Optimizer
- Query decomposition for complex questions
- Query expansion for better recall
- Type-based routing (factual, comparison, summary, etc.)

### 6. RAG Pipeline
- Hallucination prevention with confidence thresholds
- Automatic source citation
- Query caching
- History tracking

### 7. Evaluator
- Faithfulness scoring (grounding check)
- Relevance scoring
- Completeness scoring
- Session logging

## 🛡️ Hallucination Prevention

The system implements multiple strategies:

1. **Confidence Thresholds**: Rejects low-confidence retrievals
2. **Strict Prompting**: Instructs LLM to stay within context
3. **Source Citation**: Forces grounding in retrieved documents
4. **Evaluation Metrics**: Tracks faithfulness scores

## 📊 Evaluation Metrics

- **Faithfulness**: Answer grounded in context (0-1)
- **Relevance**: Answer addresses query (0-1)
- **Completeness**: Context contains answer (0-1)
- **Source Quality**: Average similarity scores
- **Overall Score**: Aggregate metric

## 🔄 Optimization Tips

### For Better Accuracy
1. Use `BAAI/bge-large-en-v1.5` embeddings (1024 dim)
2. Enable re-ranking with cross-encoder
3. Increase `min_confidence_threshold` to 0.6+
4. Use larger chunk overlap (300-400)

### For Better Speed
1. Use `all-MiniLM-L6-v2` embeddings (384 dim)
2. Disable re-ranking
3. Enable caching
4. Reduce `top_k` to 5

### For Better Recall
1. Lower `min_similarity_score` to 0.2
2. Increase `top_k` to 15
3. Enable query expansion
4. Use hybrid search with balanced alpha (0.5)

## 🚧 Roadmap

- [ ] Multi-modal support (images, tables)
- [ ] Async processing for scalability
- [ ] API server with FastAPI
- [ ] Web UI for document upload
- [ ] Support for more LLM providers (OpenAI, Anthropic)
- [ ] Advanced evaluation with RAGAS
- [ ] Multi-language support
- [ ] Contextual compression

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.

## 📧 Support

For issues or questions, please open a GitHub issue.

---

**Built with ❤️ for production RAG applications**