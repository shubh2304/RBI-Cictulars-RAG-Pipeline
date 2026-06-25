# RBI Regulatory RAG System – Under the Hood & Setup Guide

This document describes the design philosophy, technical architecture, and execution details of the Reserve Bank of India (RBI) Regulatory RAG System. The system is built from scratch without high-level RAG orchestration frameworks (no LangChain, no LlamaIndex) to ensure deep customization, complete execution control, and transparency.

---

## 1. Technical Stack

The system relies strictly on low-level libraries to handle core operations:
* **PDF Processing**: `PyMuPDF` (fast text extraction & metadata matching) and `pdfplumber` (clean tabular layout extraction).
* **OCR Fallback**: `paddleocr` (for image-only scanned files).
* **Database & Cache**: `sqlite3` (native SQL storage for parsed metadata, audit log trails, and semantic caching).
* **Lexical Search (Sparse)**: `rank-bm25` (pure python BM25Okapi implementation).
* **Semantic Search (Dense)**: `sentence-transformers` (runs `BAAI/bge-small-en-v1.5` embeddings) and `faiss-cpu` (exact inner-product indexing for cosine matches).
* **Reranking**: `sentence-transformers.CrossEncoder` (runs `BAAI/bge-reranker-base` to refine candidates).
* **Local Inference**: Any OpenAI-compatible REST server (Ollama, llama.cpp, vLLM) executing a quantized local LLM (recommended: `qwen2.5:7b-instruct`).
* **API Service**: `FastAPI` (REST endpoints, caching layers, and startup event index checks) and `uvicorn`.

---

## 2. System Architecture & How it Works

The pipeline is structured into three execution cycles: Ingestion, Retrieval, and Generation.

```
                  [1. INGESTION CYCLE]
                        PDF Files
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       Standard PDF                 Scanned PDF
      (fitz extraction)            (paddleocr raster)
             │                           │
             └─────────────┬─────────────┘
                           ▼
                 Hierarchical Parser
             (Traces Chapters / Sections)
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       Metadata Seeding            Markdown Tables
       (SQLite DB tables)          (MD format chunks)
                           │
                           ▼
                  BGE Embedding Model
                  (FAISS Flat Index)

-----------------------------------------------------------

                  [2. RETRIEVAL CYCLE]
                       User Query
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      Dense Retrieval              Sparse Retrieval
    (BGE Prompt + FAISS)           (Regex token BM25)
        (Top-50 Index)              (Top-50 Lexical)
             │                           │
             └─────────────┬─────────────┘
                           ▼
               Reciprocal Rank Fusion (RRF)
               (Fuses ranks via 1 / (60+r))
                           │
                           ▼
                   BGE Cross-Encoder
                    (Top-5 Reranked)

-----------------------------------------------------------

                  [3. GENERATION CYCLE]
                     Context + Query
                           │
                           ▼
                       Local LLM
                  (JSON Grammar Mode)
                           │
                           ▼
               Citation Verification Layer
             (Semantic Cosine + Jaccard)
                           │
                           ▼
               Validated Answer + Logs
```

### Ingestion & Chunking
1. **Salutation Heuristics**: The extractor searches the first page for recipient greetings (e.g. `Madam/Sir,`). It isolates the immediate next lines to capture clean, brief document names, discarding recipient blocks and lengthy Consolidation Appendixes.
2. **Hierarchical Traversal**: The text is stitched using page boundary tags (e.g., `[PAGE_NUM:X]`). A state machine parses lines sequentially. When it encounters header rules (`CHAPTER I`, `4.1`, `Q 1.`), it marks section boundaries, preserving active page and chapter contexts.
3. **Table Extraction**: `pdfplumber` isolates tabular grids, replacing unformatted text streams with clean Markdown table blocks (`| Col 1 | Col 2 |`). These are stored as separate table chunks.
4. **Parent-Child Linking**: Subsections (e.g., `2.2.1`) are programmatically linked to their parent containers (e.g., `2.2`) via database mappings.

### Hybrid Retrieval & Fusion
1. **BGE Instructions**: For dense search, the query is prepended with the instruction: *"Represent this question for searching relevant passages: "*; chunk documents are encoded without modifications.
2. **FAISS Search**: Query embeddings are matched against a Flat Inner Product index (`IndexFlatIP`). Because embeddings are normalized, the inner product yields exact Cosine similarity.
3. **BM25 Search**: The query is parsed using a regex tokenizer (`\b\w+\b`) and scored against database texts using BM25Okapi.
4. **Rank Fusion (RRF)**: Candidates from dense and sparse lists are merged. Chunks that rank high in both indices rise to the top, balancing exact term matches (e.g. matching circular IDs) and semantic matches.

### Reranking Layer
The top 15 hybrid candidates are passed to a Cross-Encoder reranker (`BAAI/bge-reranker-base`). Concatenating the query and passage allows the transformer to run full token-level self-attention, reordering chunks to select the top 5 most contextually relevant blocks.

### Generation & Verified Citations
1. **JSON Grammar Mode**: The local LLM is instructed via system prompt and API configurations to return a valid JSON object matching a strict format schema:
   ```json
   {
     "response": "The limit is raised to Rs. 3 Lakhs [1].",
     "citations": [
       { "citation_tag": "[1]", "source_statement": "limit raised to Rs. 3 Lakhs", "source_block_index": 1 }
     ]
   }
   ```
2. **Verification layer**: Each citation's `source_statement` is semantically encoded and cosine-compared to the actual source block text. If the similarity is lower than `0.70`, the system alerts the user and flags the citation, preventing LLM hallucinations from polluting official queries.
3. **Caching Layer**: Queries are cached in SQLite using exact string matches. A query cache hit returns the structured response in sub-milliseconds without firing neural models.

---

## 3. Project File Layout

```
RBI RAG/
├── main.py                   # CLI RAG pipeline entrypoint
├── api_server.py             # FastAPI REST Server (serves query API & PDF streams)
├── evaluate.py               # Evaluation Suite (Recall@K, MRR accuracy calculations)
├── rbi_rag.db                # SQLite database (populated dynamically)
├── faiss_index.index         # Vector index binary
├── embeddings.npy            # NumPy embedding matrix backup
│
├── database/
│   ├── connection.py         # SQLite connection manager
│   └── models.py             # Schema declarations & DB initializer
│
├── ingestion/
│   ├── pdf_extractor.py      # Fitz + pdfplumber page/table reader & title classifier
│   ├── ocr_fallback.py       # Scanned PDF text extraction via paddleocr
│   ├── parser.py             # Section-aware parser & parent-child chunk linker
│   └── seed.py               # Document scanning & DB populator
│
├── retrieval/
│   ├── dense.py              # BGE small embedding service & FAISS searcher
│   ├── sparse.py             # BM25 token index searcher
│   ├── fusion.py             # Reciprocal Rank Fusion (RRF) combiner
│   └── reranker.py           # Cross-encoder candidate scorer
│
├── generation/
│   ├── llm_client.py         # Local chat endpoint client (Ollama/llama.cpp / Native CPU fallback)
│   └── citation_verifier.py  # Lexical Jaccard & semantic validation engine
│
└── frontend/                 # Next.js App Router Web Application
    ├── package.json          # Node dependencies
    ├── src/
    │   └── app/
    │       ├── page.tsx      # Multi-pane dashboard (Library Sidebar, Chat Panel, PDF Viewer)
    │       ├── globals.css   # Tailored dark-mode variables, animations, scrollbars
    │       └── layout.tsx    # App shell wrapper
    └── tailwind.config.ts    # Styling overrides
```

---

## 4. Operational Runbook

### Prerequisites
Install all backend dependencies:
```powershell
pip install uvicorn fastapi pydantic pymupdf pdfplumber sentence-transformers faiss-cpu rank-bm25 torch paddlepaddle paddleocr
```

Ensure Node.js is installed for running the frontend dashboard client.

### 1. Database Seeding & Ingestion
Ensure your RBI documents are in the `circulars/` directory and run:
```powershell
python -m ingestion.seed
```
*This scans, parses, builds the FAISS indexes, and updates your SQLite file.*

### 2. Startup local LLM Server (Optional)
Start Ollama with the recommended model:
```powershell
ollama run qwen2.5:7b-instruct
```
*(If Ollama is offline or unavailable, the system automatically falls back to running a local Qwen 0.5B model in-memory on CPU)*

### 3. Launching the FastAPI Backend API
Start the FastAPI server:
```powershell
uvicorn api_server:app --reload --port 8000
```
*This hosts the REST API endpoints and streams PDF files on `http://localhost:8000`.*

### 4. Launching the Next.js Frontend Dashboard
Navigate to the `frontend` folder and start the development server:
```powershell
cd frontend
npm run dev
```
*Open your browser and navigate to `http://localhost:3000` to interact with the visual dashboard.*

### 5. Running Benchmark Audits
Run the retrieval performance verification suite:
```powershell
python -m evaluate
```

