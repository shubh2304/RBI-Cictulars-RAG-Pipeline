import time
import uuid
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from database.connection import get_connection
from ingestion.seed import seed_database
from retrieval.dense import IndexManager
from retrieval.fusion import HybridRetriever
from retrieval.reranker import Reranker
from generation.llm_client import LLMClient
from generation.citation_verifier import CitationVerifier

app = FastAPI(
    title="RBI Regulatory RAG System API",
    description="Enterprise-grade RAG engine forReserve Bank of India (RBI) documents, built from first principles.",
    version="1.0.0"
)

# Initialize retrievers lazily or on startup
retriever = None

@app.on_startup
def startup_event():
    global retriever
    # Pre-build index on startup if not already done
    try:
        IndexManager.build_and_save_index()
    except Exception as e:
        print(f"Startup warning - could not build vector index: {e}")
    retriever = HybridRetriever()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    bypass_cache: bool = False

@app.post("/query")
def run_query(request: QueryRequest):
    """
    Solves user queries against the RBI document corpus.
    Uses Semantic caching, Hybrid search, Cross-Encoder Reranking, Qwen2.5-7B synthesis, and citation verification.
    """
    global retriever
    start_time = time.time()
    
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
        
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Caching Layer Check
    if not request.bypass_cache:
        cursor.execute("SELECT response_text FROM semantic_cache WHERE query_text = ?", (query.lower(),))
        cached_row = cursor.fetchone()
        if cached_row:
            execution_time = (time.time() - start_time) * 1000
            print(f"Cache hit! Query resolved in {execution_time:.2f}ms")
            conn.close()
            # Return cached response (which was stored as JSON)
            return json.loads(cached_row["response_text"])
            
    # 2. Hybrid Retrieval (BM25 + FAISS + RRF Score Fusion)
    if retriever is None:
        retriever = HybridRetriever()
    
    # Fetch top 15 candidates for reranking
    candidates = retriever.search(query, top_k=15)
    
    if not candidates:
        response = {
            "response": "No relevant RBI guidelines or notifications were found in the database.",
            "citations": [],
            "warnings": ["No context found."],
            "execution_time_ms": (time.time() - start_time) * 1000
        }
        conn.close()
        return response

    # 3. Reranking Layer (BGE-Reranker Base)
    reranked_chunks = Reranker.rerank(query, candidates, top_k=request.top_k)
    
    # 4. LLM Response Generation (quantized Qwen2.5-7B)
    llm_output = LLMClient.generate_answer(query, reranked_chunks)
    
    # 5. Citation Verification Engine
    final_response = CitationVerifier.verify_citations(llm_output, reranked_chunks)
    
    # Format execution metadata
    execution_time_ms = (time.time() - start_time) * 1000
    final_response["execution_time_ms"] = round(execution_time_ms, 2)
    
    # 6. Save to Caching and Audit Log Tables
    try:
        response_json = json.dumps(final_response)
        
        # Insert cache entry (lowercased key)
        cursor.execute(
            "INSERT OR REPLACE INTO semantic_cache (cache_id, query_text, response_text) VALUES (?, ?, ?)",
            (str(uuid.uuid4()), query.lower(), response_json)
        )
        
        # Insert audit log entry
        cursor.execute(
            "INSERT INTO query_logs (log_id, query_text, response_text, execution_time_ms) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), query, response_json, execution_time_ms)
        )
        conn.commit()
    except Exception as e:
        print(f"DB Log/Cache error: {e}")
    finally:
        conn.close()
        
    return final_response


@app.get("/documents")
def list_documents():
    """Lists all RBI PDFs currently registered in the system."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT document_id, filename, document_name, document_type, ref_number, circular_number, pub_date, source_url
        FROM documents
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.post("/ingest")
def trigger_ingestion():
    """Triggers scan and ingestion of new documents in the circulars/ directory."""
    try:
        seed_database()
        IndexManager.build_and_save_index()
        # Re-initialize sparse retriever corpus to load new chunks
        global retriever
        retriever = HybridRetriever()
        return {"status": "success", "message": "Corpus scan and index rebuild completed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """Service health state check."""
    return {"status": "healthy", "timestamp": time.time()}
