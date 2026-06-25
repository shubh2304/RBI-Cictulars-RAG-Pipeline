# RBI RAG System – Query-to-Response Lifecycle Explainer

This document outlines the step-by-step lifecycle of a compliance query in the RBI Regulatory RAG System, from the moment a user submits a question to the printing of the semantically verified response in the terminal.

---

## 1. Summary of the Lifecycle

```
[User Query] 
     │
     ▼
Step 1: Input & Formatting ── Prepend BGE prompt & Tokenize query
     │
     ├──────────────────────────────────────┐
     ▼                                      ▼
Step 2a: Dense Retrieval (FAISS)      Step 2b: Sparse Retrieval (BM25)
  (Computes Cosine Similarity)          (Computes TF-IDF term overlaps)
  (Returns top-50 matches)              (Returns top-50 matches)
     │                                      │
     └──────────────────┬───────────────────┘
                        ▼
Step 3: Reciprocal Rank Fusion (RRF) ── Merges candidates by rank scores
                        │ (Top 30 Chunks)
                        ▼
Step 4: Cross-Encoder Reranker ─────── Full self-attention scoring via BGE
                        │ (Top 5 Chunks)
                        ▼
Step 5: LLM Generation ─────────────── Chat completions (Ollama or native CPU fallback)
                        │ (JSON Output)
                        ▼
Step 6: Citation Verification ──────── Lexical Jaccard & Semantic Cosine check
                        │ (Verified Footnotes)
                        ▼
Step 7: Server Log & Audit ──────────── Log queries, verify token boundaries, cache hits
                        │ (FastAPI CORS Response)
                        ▼
Step 8: Web Dashboard Display ──────── Render Markdown bubble & mount PDF at cited page (#page=N)
     │
     ▼
[Final Dashboard Split-Pane View]
```

---

## 2. Detailed Step-by-Step Walkthrough

### Step 1: Input Preprocessing & Formatting
When you enter a query (e.g. *"What is the collateral-free limit for agricultural credit?"*), the system prepares the query for two different search engines:
1. **For Dense Search**: The BGE model (`bge-small-en-v1.5`) requires a specific search instruction to perform optimally. The system prepends the instruction:
   `"Represent this question for searching relevant passages: What is the collateral-free limit for agricultural credit?"`
2. **For Sparse Search**: The system splits the raw text into a list of lowercase alphanumeric words using a custom regex tokenizer (`\b\w+\b`):
   `['what', 'is', 'the', 'collateral', 'free', 'limit', 'for', 'agricultural', 'credit']`

---

### Step 2: Parallel Dual Retrieval
The query is searched simultaneously across two indices:

#### A. Dense Retrieval (FAISS)
* The query is embedded into a **384-dimension** floating-point vector using our cached SentenceTransformer model.
* The vector is searched against the pre-compiled `faiss_index.index` file (which stores embeddings of all 335 chunks in the database).
* Since the index uses Inner Product matching (`faiss.IndexFlatIP`) and vectors are normalized, this returns exact **Cosine Similarity** values. The retriever fetches the **Top-50** candidate chunks.

#### B. Sparse Retrieval (BM25)
* The tokenized word list is matched against our tokenized document corpus index using the **BM25Okapi** algorithm.
* This scores documents based on term frequency and inverse document frequency (matching specific keywords like `collateral-free` and `agricultural`). The retriever returns the **Top-50** candidate chunks.

---

### Step 3: Reciprocal Rank Fusion (RRF)
The dense and sparse lists are combined using Reciprocal Rank Fusion (RRF). RRF ranks chunks by calculating a score based on their position in both search lists:
$$RRF(chunk) = \frac{1}{60 + \text{Rank}_{\text{dense}}} + \frac{1}{60 + \text{Rank}_{\text{sparse}}}$$
* If a chunk is in the Top-3 of both searches, its rank score is extremely high.
* If it only appears in one list, it is still retained but given a lower priority.
* The system merges the list and keeps the **Top-30** highest scoring chunks.

---

### Step 4: Cross-Encoder Reranking
Dense and sparse retrieval search document spaces independently (bi-encoder search). To determine exact alignment, the query and candidates must be cross-analyzed.
* The Top-30 candidates are passed to a Cross-Encoder reranker (`BAAI/bge-reranker-base`).
* The system concatenates the query and each chunk into a single sequence: `[Query, Chunk Text]`.
* The transformer runs full self-attention across all tokens in both strings, scoring their literal relevance.
* The system sorts the list and retains the **Top-5** chunks.

---

### Step 5: Local LLM Generation & Structured Formatting
The top chunks are formatted as context blocks:
```text
--- CONTEXT BLOCK [1] ---
Source: Credit Flow to Agriculture... | Page: 2 | Section: None
Content: The collateral-free agricultural loan limit has been raised...
-------------------------
```
The query and formatted context are sent to the local LLM with system instructions enforcing JSON output:
1. **OpenAI API Check**: Attempts to connect to a local Ollama server.
2. **CPU Fallback**: If the server is offline, the client loads the native python transformers model `Qwen2.5-0.5B-Instruct` in-memory.
3. The LLM runs text generation and returns a JSON string containing the response paragraphs and list of citation tags.

---

### Step 6: Citation Verification Engine
To prevent hallucinated citations, the system executes double-check validation:
1. **Lexical Jaccard Overlap**: Calculates word set similarities between the generated statement and actual source chunk.
2. **Semantic Cosine Similarity**: Embeds the statement and source chunk, calculating vector cosine alignment.
3. **Verification**: If the similarity is $\ge 0.70$ (or Jaccard $\ge 0.40$), the citation is marked as `[VERIFIED]` and enriched with document name, page number, ref number, and section title. If not, it flags a `[HALLUCINATION WARNING]`.

---

### Step 7: Server Response & Audit Trail
Before returning the JSON payload to the client, the server performs audit and optimization routines:
* **Log Queries**: Writes the query text, execution latency, response metadata, and verification verdicts to the SQLite database `query_logs` table for compliance tracking.
* **CORS Header Inclusion**: Appends headers allowing cross-origin resource requests from the Next.js dev server port (e.g., origin `http://localhost:3000`).

---

### Step 8: Web Dashboard Display & Page Redirection
Once the JSON response lands on the Next.js client:
1. **Citation Parsing**: A regular expression scans the flowing response text for citation tags matching the format `[N: Document, Page X]`.
2. **Inline Citation Button Mounting**: Replaces the plain text strings with interactive buttons. Clicking these buttons updates the active PDF state.
3. **Double-Pane Display**: Renders the text response on the left, and lists verified vs. hallucinated source text snippets below the bubble.
4. **Target Page Jump**: If the user clicks "View source page" or an inline citation button, the PDF viewer on the right mounts an iframe pointing to:
   `${BACKEND_URL}/api/document/${filename}#page=${page_number}`
   To resolve browser iframe cache anchor issues, a state-driven key `key={viewerKey}` is bound to the iframe and incremented on every selection, forcing the browser to reload and focus on the exact cited page.
