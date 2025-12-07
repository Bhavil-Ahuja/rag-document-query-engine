# RAG Document Query Engine
A production-ready RAG backend for intelligent Q&amp;A over PDF/doc collections. Supports ingestion, text extraction, chunking, Pinecone vector indexing, semantic search, context assembly, and LLM-based answers with caching and token analytics. Built with Spring Boot for scalable delivery.

## 🚀 Features

- Document Ingestion – PDF/DOCX upload + extraction
- Text Cleaning & Chunking – configurable window + overlap
- Vector Indexing – Pinecone for scalable embedding search
- Semantic Retrieval – top-k similarity search
- LLM Answer Generation – context-aware, grounded responses
- Caching Layer – reduced latency + cost optimization
- Token Analytics – track usage, latency, and costs
- Metadata Storage – Postgres for documents/chunks
- Modular Architecture – clean, extensible Spring Boot design

## 📐 High-Level Architecture

```
A[User Query] --> B[Query Embedding]
B --> C[Pinecone Vector Search]
C --> D[Fetch Chunk Metadata (Postgres)]
D --> E[Context Builder]
E --> F[LLM Generation]
F --> G[Cache Store]
G --> H[Final Answer]
```

## 📁 Project Structure

```
rag-document-query-engine/
 ├─ src/main/java/com/rag/engine/
 │   ├─ controller/
 │   ├─ service/
 │   ├─ rag/
 │   ├─ embedding/
 │   ├─ vectorstore/
 │   ├─ llm/
 │   ├─ model/
 │   └─ config/
 ├─ src/main/resources/
 │   ├─ application.yml
 ├─ README.md
 └─ pom.xml
```

 ## 🔧 Tech Stack

- Java 21, Spring Boot
- Pinecone (vector DB)
- OpenAI / Claude (LLM APIs)
- Postgres (metadata + chunks)
- Redis (cache)

## 🔌 API Endpoints

### Documents
- POST /documents/upload
- POST /documents/index
- GET  /documents

### Query
- POST /query

## ⚙️ Running the Project

```
git clone https://github.com/<username>/rag-document-query-engine
cd rag-document-query-engine
./mvnw clean install
./mvnw spring-boot:run
```

### Create .env:

```
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_ENV=...
DATABASE_URL=...
```
