import time
import uuid
import json
import os
import secrets
import hashlib
import logging
from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
from database.connection import get_connection
from ingestion.seed import seed_database
from retrieval.dense import IndexManager
from retrieval.fusion import HybridRetriever
from retrieval.reranker import Reranker
from generation.llm_client import LLMClient
from generation.citation_verifier import CitationVerifier

# Initialize FastAPI
app = FastAPI(
    title="RBI Regulatory RAG System API",
    description="Enterprise-grade RAG engine for Reserve Bank of India (RBI) documents, built from first principles.",
    version="1.0.0"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("security_audit")

# Secure Headers Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# CORS configuration: restrict origins to trusted sources
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "https://trusted-domain.com"], # Strict domain list for ISO compliance
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global Exception Handler to prevent information leakage / stack trace disclosure
@app.exception_handler(Exception)
def global_exception_handler(request, exc):
    logger.error(f"Internal system error occurred: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please contact the system administrator."}
    )

# Retrieve or generate API Keys on startup
RAG_API_KEY = os.getenv("RAG_API_KEY")
RAG_ADMIN_API_KEY = os.getenv("RAG_ADMIN_API_KEY")

if not RAG_API_KEY:
    RAG_API_KEY = f"user_fallback_{secrets.token_hex(16)}"
    logger.warning(f"RAG_API_KEY not set. Generated secure user API key: {RAG_API_KEY}")

if not RAG_ADMIN_API_KEY:
    RAG_ADMIN_API_KEY = f"admin_fallback_{secrets.token_hex(16)}"
    logger.warning(f"RAG_ADMIN_API_KEY not set. Generated secure admin API key: {RAG_ADMIN_API_KEY}")

# Compute SHA256 hashes of keys for secure comparison
def compute_hash(val: str) -> str:
    return hashlib.sha256(val.encode("utf-8")).hexdigest()

USER_KEY_HASH = compute_hash(RAG_API_KEY)
ADMIN_KEY_HASH = compute_hash(RAG_ADMIN_API_KEY)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def authenticate_client(api_key: str = Security(API_KEY_HEADER)):
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key is missing from X-API-Key header."
        )
    key_hash = compute_hash(api_key)
    if key_hash == ADMIN_KEY_HASH:
        return {"role": "admin", "identity": "admin_client_" + key_hash[:8]}
    elif key_hash == USER_KEY_HASH:
        return {"role": "user", "identity": "user_client_" + key_hash[:8]}
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key."
        )

def require_admin(client: dict = Depends(authenticate_client)):
    if client["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for this operation."
        )
    return client

# Initialize retrievers lazily or on startup
retriever = None

@app.on_event("startup")
def startup_event():
    global retriever
    # Pre-build index on startup if not already done
    try:
        IndexManager.build_and_save_index()
    except Exception as e:
        logger.warning(f"Startup warning - could not build vector index: {e}")
    retriever = HybridRetriever()

class QueryRequest(BaseModel):
    query: str = Field(..., max_length=1000, description="Compliance search query (max 1000 characters)")
    top_k: int = Field(5, ge=1, le=50, description="Retrieve top K candidates (1 to 50)")
    bypass_cache: bool = False

@app.post("/query")
def run_query(request: QueryRequest, client: dict = Depends(authenticate_client)):
    """
    Solves user queries against the RBI document corpus.
    Requires Standard or Admin API Key. Enforces input size validation, CORS, 
    exception sanitization, and auditable client tagging.
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
            logger.info(f"Cache hit for query by client '{client['identity']}' resolved in {execution_time:.2f}ms")
            conn.close()
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
        
        # Insert audit log entry attributing activity to validated client identifier
        cursor.execute(
            "INSERT INTO query_logs (log_id, user_identity, query_text, response_text, execution_time_ms) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), client["identity"], query, response_json, execution_time_ms)
        )
        conn.commit()
    except Exception as e:
        logger.error(f"DB Log/Cache error: {e}")
    finally:
        conn.close()
        
    return final_response


@app.get("/documents")
def list_documents(client: dict = Depends(authenticate_client)):
    """Lists all RBI PDFs currently registered in the system. Requires API authentication."""
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
def trigger_ingestion(client: dict = Depends(require_admin)):
    """Triggers scan and ingestion of new documents in the circulars/ directory. Requires Admin privileges."""
    try:
        seed_database()
        IndexManager.build_and_save_index()
        # Re-initialize sparse retriever corpus to load new chunks
        global retriever
        retriever = HybridRetriever()
        logger.info(f"Ingestion triggered and completed by administrator '{client['identity']}'")
        return {"status": "success", "message": "Corpus scan and index rebuild completed."}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Ingestion process encountered an error and could not complete."
        )


@app.get("/health")
def health_check():
    """Public health state check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}
