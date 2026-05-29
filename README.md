# Multimodal RAG — Production-Grade System

A complete Retrieval-Augmented Generation system supporting **text and images**, built with
Milvus, Chainlit, CLIP, BGE-M3, and GPT-4o.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       USER INTERFACE                            │
│                  Chainlit (port 8080)                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
           ┌────────────────▼────────────────┐
           │         CACHE LAYER             │
           │   Redis: Exact (MD5) +          │
           │   Semantic (cosine > 0.92)      │
           └────────────────┬────────────────┘
                            │ cache miss
           ┌────────────────▼────────────────┐
           │       RETRIEVAL PIPELINE        │
           │                                 │
           │  Query → BGE-M3 encode          │
           │         ┌──────┴──────┐         │
           │   Dense Search  Sparse Search   │
           │   (HNSW/COSINE) (BM25/IP)       │
           │         └──────┬──────┘         │
           │           RRF Fusion            │
           │         (pymilvus RRFRanker)    │
           │                │                │
           │    Cross-Encoder Reranker       │
           │  (ms-marco-MiniLM-L-6-v2)       │
           └────────────────┬────────────────┘
                            │
           ┌────────────────▼────────────────┐
           │       GENERATION (LLM)          │
           │  GPT-4o (streaming) /           │
           │  LLaVA fallback (Ollama)        │
           └────────────────┬────────────────┘
                            │
           ┌────────────────▼────────────────┐
           │        RESPONSE + SOURCES       │
           │  Streamed answer with citations │
           └─────────────────────────────────┘

INGESTION PATH
──────────────
  PDF / PNG / JPG / TXT
        │
   PyMuPDF / Pillow
        │
   Text chunks (BGE-M3 dense+sparse)
   Image chunks (CLIP 512d → padded 1024d)
        │
   Milvus Collection: multimodal_rag
   ┌────────────────────────────────┐
   │ id │ doc_id │ modality │ ...  │
   │ dense_vector (1024d HNSW)     │
   │ sparse_vector (SPARSE_INV)    │
   │ metadata (JSON)               │
   └────────────────────────────────┘
```

---

## Prerequisites

- **Docker** ≥ 24 + **Docker Compose** ≥ 2.20
- **Python** 3.11
- **Poetry** ≥ 1.8  (`pip install poetry`)
- **OpenAI API key** (or local Ollama + LLaVA for the free path)

---

## Quick Start

### 1. Clone and configure

```bash
git clone <repo-url> multimodal-rag
cd multimodal-rag
cp .env.example .env
# Edit .env: set OPENAI_API_KEY (or leave blank for Ollama fallback)
```

### 2. Start infrastructure

```bash
cd docker
docker compose up -d
# Wait ~30s for Milvus to initialise
docker compose ps          # all services should be "healthy"
```

Services exposed:
| Service | URL |
|---------|-----|
| Chainlit UI | http://localhost:8080 |
| Attu (Milvus GUI) | http://localhost:8000 |
| MinIO console | http://localhost:9001 |
| Redis | localhost:6379 |
| Milvus | localhost:19530 |

### 3. Install Python dependencies

```bash
poetry install
```

### 4. Ingest sample documents

```bash
# Place PDFs / images in data/raw/
poetry run python scripts/ingest.py --path data/raw/
```

### 5. Launch the chatbot

```bash
poetry run chainlit run app/chainlit_app.py --host 0.0.0.0 --port 8080
```

Open http://localhost:8080 in your browser.

---

## CLI Commands

```bash
# Ingest a single file
poetry run python scripts/ingest.py --path data/raw/report.pdf

# Ingest a directory
poetry run python scripts/ingest.py --path data/raw/

# Reset collection and re-ingest
poetry run python scripts/ingest.py --path data/raw/ --reset

# Run RAGAS evaluation (20 Q&A pairs)
poetry run python scripts/evaluate.py

# Run evaluation with fewer samples
poetry run python scripts/evaluate.py --samples 5

# Run tests
poetry run pytest
```

---

## Project Structure

```
multimodal-rag/
├── app/
│   ├── chainlit_app.py          # Chatbot UI entry point
│   ├── ingestion/               # PDF + image loading & chunking
│   ├── embeddings/              # CLIP encoder, BGE-M3 encoder
│   ├── vectorstore/             # Milvus client (CRUD + hybrid search)
│   ├── retrieval/               # Hybrid retriever + cross-encoder reranker
│   ├── generation/              # LangChain chain (GPT-4o / LLaVA)
│   ├── cache/                   # Redis exact + semantic cache
│   └── utils/                   # Config (pydantic-settings), loguru logger
├── evaluation/
│   └── ragas_eval.py            # RAGAS faithfulness/relevancy/recall/precision
├── scripts/
│   ├── ingest.py                # CLI batch ingestion
│   └── evaluate.py              # CLI evaluation runner
├── tests/                       # pytest test suite (mocked infra)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml       # Milvus + Redis + Attu + App
└── data/
    ├── raw/                     # Input documents
    └── processed/               # Post-ingestion artifacts
```

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | GPT-4o API key (leave blank for Ollama) |
| `MILVUS_HOST` | `localhost` | Milvus host |
| `MILVUS_PORT` | `19530` | Milvus port |
| `MILVUS_COLLECTION` | `multimodal_rag` | Collection name |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |
| `REDIS_TTL` | `3600` | Cache TTL in seconds |
| `CACHE_SIMILARITY_THRESHOLD` | `0.92` | Cosine threshold for semantic cache |
| `EMBEDDING_MODEL_TEXT` | `BAAI/bge-m3` | BGE-M3 model |
| `EMBEDDING_MODEL_IMAGE` | `openai/clip-vit-base-patch32` | CLIP model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder |
| `TOP_K_RETRIEVAL` | `20` | Candidates retrieved before reranking |
| `TOP_K_RERANK` | `5` | Final results after reranking |
| `CHUNK_SIZE` | `512` | Text chunk size in characters |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `text` | `text` (dev) or `json` (production) |

---

## Key Design Decisions

- **Singleton model loading**: CLIP, BGE-M3, and the cross-encoder are loaded once at startup.
  They are thread-safe and shared across requests to avoid repeated ~10s load times.
- **Idempotent ingestion**: Before inserting, existing entries for the same `doc_id` are deleted.
  Re-ingesting a document is safe.
- **Hybrid search**: Dense HNSW (semantic similarity) + Sparse BM25 (keyword matching) fused
  with RRF ensures both semantic and lexical relevance.
- **Two-layer cache**: MD5 exact match catches identical questions in O(1). Semantic cache
  catches paraphrases (cosine ≥ 0.92) to avoid redundant LLM calls.
- **Tenacity retries**: All Milvus and Redis operations have automatic exponential backoff.

---

## License

MIT
