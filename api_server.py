import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sqlite3

# Import our RAG modules
from database.connection import get_connection
from retrieval.fusion import HybridRetriever
from retrieval.reranker import Reranker
from generation.llm_client import LLMClient
from generation.citation_verifier import CitationVerifier

# Initialize FastAPI app
app = FastAPI(title="RBI Compliance RAG API", description="Backend server for querying RBI circulars RAG pipeline")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global retriever instance to avoid building BM25/loading models on every query
print("Initializing Hybrid Retriever at server startup...")
hybrid_retriever = HybridRetriever()
print("Pre-loading Embedding model at server startup...")
from retrieval.dense import EmbeddingService
EmbeddingService.get_model()
print("Pre-loading Reranker model at server startup...")
from retrieval.reranker import Reranker
Reranker.get_model()
print("Server RAG components loaded successfully.")

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
def query_rag(req: QueryRequest):
    """
    Exposes the complete RAG query pipeline.
    Retrieves, reranks, generates an answer via LLM, and verifies citations.
    """
    query_text = req.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
        
    try:
        print(f"\n--- Processing query: '{query_text}' ---")
        
        # Check for simple greetings or smalltalk to respond instantly
        greeting_res = LLMClient.check_greetings_and_smalltalk(query_text)
        if greeting_res:
            print(f"Intercepted greeting/smalltalk query: '{query_text}'")
            return greeting_res
            
        # Decompose query if it's compound / multi-query
        sub_queries = LLMClient.decompose_query(query_text)
        print(f"Decomposed into {len(sub_queries)} queries: {sub_queries}")
        
        merged_chunks = []
        seen_chunk_ids = set()
        
        for idx, sub_q in enumerate(sub_queries):
            print(f"Retrieving and reranking for sub-query [{idx+1}/{len(sub_queries)}]: '{sub_q}'")
            # Fetch candidates for each sub-query
            candidates = hybrid_retriever.search(sub_q, top_k=5)
            if not candidates:
                continue
            # Rerank candidates against this sub-query
            reranked_sub = Reranker.rerank(sub_q, candidates, top_k=5)
            for chunk in reranked_sub:
                c_id = chunk["chunk_id"]
                if c_id not in seen_chunk_ids:
                    seen_chunk_ids.add(c_id)
                    merged_chunks.append(chunk)
                else:
                    # If already present, keep the one with higher rerank score
                    for existing in merged_chunks:
                        if existing["chunk_id"] == c_id:
                            if chunk.get("rerank_score", 0.0) > existing.get("rerank_score", 0.0):
                                existing["rerank_score"] = chunk["rerank_score"]
                            break
                            
        # Sort by rerank score descending and limit to top results
        merged_chunks.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        final_chunks = merged_chunks[:15]
        
        if not final_chunks or final_chunks[0].get("rerank_score", -99.0) < 0.02:
            print("No relevant chunks found or top chunk relevance score too low. Query is likely out-of-scope.")
            return {
                "response": "The query is outside the scope of the ingested RBI regulatory guidelines. I am only trained to answer questions about RBI compliance and circulars.",
                "citations": [],
                "warnings": ["Out of scope query."],
                "hallucination_detected": False,
                "answerable": False
            }
            
        # 3. LLM Generation
        print("Generating answer from LLM...")
        llm_output = LLMClient.generate_answer(query_text, final_chunks, sub_queries=sub_queries)
        
        # 4. Citation Verification
        print("Verifying citations...")
        final_response = CitationVerifier.verify_citations(llm_output, final_chunks)
        print("Query processing completed successfully!")
        
        return final_response
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing query: {str(e)}")

@app.get("/api/documents")
async def list_documents():
    """
    Returns a list of all ingested circulars stored in the SQLite database.
    """
    try:
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT filename, document_name, document_type, ref_number, circular_number, pub_date, source_url 
            FROM documents 
            ORDER BY pub_date DESC
        """)
        rows = cur.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/document/{filename}")
async def get_pdf_document(filename: str):
    """
    Streams a requested PDF file from the circulars/ directory.
    This enables split-pane PDF rendering in the browser with fragment page support (#page=N).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    circulars_dir = os.path.abspath(os.path.join(base_dir, "circulars"))
    safe_path = os.path.abspath(os.path.join(circulars_dir, filename))
    
    # Directory traversal defense
    if not safe_path.startswith(circulars_dir):
        raise HTTPException(status_code=400, detail="Invalid file path.")
        
    if not os.path.exists(safe_path):
        raise HTTPException(status_code=404, detail="Requested PDF document not found.")
        
    return FileResponse(safe_path, media_type="application/pdf")

@app.get("/api/health")
async def health_check():
    """Simple API health check endpoint."""
    return {"status": "ok", "retriever": "initialized", "index_size": len(hybrid_retriever.sparse_retriever.chunks)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
