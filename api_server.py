from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
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
print("Server RAG components loaded successfully.")

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
async def query_rag(req: QueryRequest):
    """
    Exposes the complete RAG query pipeline.
    Retrieves, reranks, generates an answer via LLM, and verifies citations.
    """
    query_text = req.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
        
    try:
        # 1. Retrieve candidates
        candidates = hybrid_retriever.search(query_text, top_k=30)
        if not candidates:
            return {
                "response": "The provided RBI circulars do not contain sufficient information to answer this query.",
                "citations": [],
                "warnings": [],
                "hallucination_detected": False,
                "answerable": False
            }
            
        # 2. Rerank
        reranked = Reranker.rerank(query_text, candidates, top_k=5)
        
        # 3. LLM Generation
        llm_output = LLMClient.generate_answer(query_text, reranked)
        
        # 4. Citation Verification
        final_response = CitationVerifier.verify_citations(llm_output, reranked)
        
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
