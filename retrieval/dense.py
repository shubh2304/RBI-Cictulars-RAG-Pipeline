import os
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from database.connection import get_connection

# Configuration
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this question for searching relevant passages: "
EMBED_DIM = 384  # bge-small-en-v1.5 dimension

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.index")
EMB_PATH = os.path.join(BASE_DIR, "embeddings.npy")

class EmbeddingService:
    """Singleton service to load the SentenceTransformer model once."""
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            print(f"Loading embedding model: {MODEL_NAME}...")
            # Automatically runs on CUDA if GPU is available, else CPU
            cls._model = SentenceTransformer(MODEL_NAME)
            print("Model loaded successfully.")
        return cls._model

    @classmethod
    def embed_queries(cls, queries):
        """Embeds queries by prepending BGE instruction."""
        model = cls.get_model()
        prepared = [QUERY_INSTRUCTION + q for q in queries]
        return model.encode(prepared, normalize_embeddings=True)

    @classmethod
    def embed_passages(cls, passages):
        """Embeds text chunks without instructions."""
        model = cls.get_model()
        return model.encode(passages, normalize_embeddings=True, show_progress_bar=True)


class IndexManager:
    """Manages creation, loading, and updating of the dense vector indexes."""
    
    @staticmethod
    def build_and_save_index():
        """
        Fetches all unindexed chunks from DB, generates embeddings,
        updates the database chunk linkages, and creates/saves FAISS & NumPy indexes.
        """
        conn = get_connection()
        cursor = conn.cursor()
        
        # 1. Fetch chunks that need indexing
        # To handle incremental updates or complete build:
        cursor.execute("SELECT chunk_id, chunk_text FROM chunks WHERE vector_index IS NULL")
        unindexed_rows = cursor.fetchall()
        
        if not unindexed_rows:
            print("All chunks are already indexed.")
            conn.close()
            return
            
        print(f"Found {len(unindexed_rows)} unindexed chunks. Generating embeddings...")
        
        chunk_ids = [r["chunk_id"] for r in unindexed_rows]
        texts = [r["chunk_text"] for r in unindexed_rows]
        
        # Generate embeddings
        new_embs = EmbeddingService.embed_passages(texts)
        
        # Check if existing index exists
        if os.path.exists(EMB_PATH) and os.path.exists(INDEX_PATH):
            print("Loading existing embeddings and index to append...")
            existing_embs = np.load(EMB_PATH)
            all_embs = np.vstack([existing_embs, new_embs])
            index = faiss.read_index(INDEX_PATH)
        else:
            print("Creating new embeddings array and FAISS index...")
            all_embs = new_embs
            # Flat Inner Product index for exact Cosine similarity (since embs are normalized)
            index = faiss.IndexFlatIP(EMBED_DIM)
            
        # Update database with sequential vector indices
        start_idx = len(all_embs) - len(new_embs)
        for i, chunk_id in enumerate(chunk_ids):
            cursor.execute("UPDATE chunks SET vector_index = ? WHERE chunk_id = ?", (start_idx + i, chunk_id))
            
        conn.commit()
        conn.close()
        
        # Reset FAISS index and add all embeddings to keep mappings aligned
        index = faiss.IndexFlatIP(EMBED_DIM)
        index.add(all_embs.astype('float32'))
        
        # Save indexes
        np.save(EMB_PATH, all_embs)
        faiss.write_index(index, INDEX_PATH)
        print(f"Index successfully updated. Total indexed vectors: {index.ntotal}")

    @staticmethod
    def get_faiss_index():
        """Loads and returns the FAISS index."""
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError("FAISS index file not found. Run build_and_save_index() first.")
        return faiss.read_index(INDEX_PATH)

    @staticmethod
    def get_embeddings_matrix():
        """Loads and returns the embeddings matrix."""
        if not os.path.exists(EMB_PATH):
            raise FileNotFoundError("Embeddings matrix file not found. Run build_and_save_index() first.")
        return np.load(EMB_PATH)


class DenseRetriever:
    """Implements Vector search algorithms (NumPy and FAISS)."""

    def __init__(self):
        # Build index if it doesn't exist
        if not os.path.exists(INDEX_PATH) or not os.path.exists(EMB_PATH):
            IndexManager.build_and_save_index()
        
        self.faiss_index = IndexManager.get_faiss_index()
        self.embs_matrix = IndexManager.get_embeddings_matrix()

    def search_numpy(self, query, top_k=5):
        """Version 1: Pure NumPy Cosine Similarity dense search."""
        query_vector = EmbeddingService.embed_queries([query])  # Shape: (1, dim)
        
        # Dot product of normalized vectors yields exact cosine similarity
        similarities = np.dot(self.embs_matrix, query_vector.T).squeeze()
        
        # Get top-K indices sorted in descending order of similarity
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Retrieve chunk metadata from database
        results = []
        conn = get_connection()
        cursor = conn.cursor()
        
        for idx in top_indices:
            score = float(similarities[idx])
            cursor.execute("""
                SELECT c.chunk_id, c.document_id, c.parent_chunk_id, c.chunk_type, c.page_number, 
                       c.chapter_title, c.section_title, c.chunk_text, d.document_name, d.circular_number, d.ref_number
                FROM chunks c
                JOIN documents d ON c.document_id = d.document_id
                WHERE c.vector_index = ?
            """, (int(idx),))
            row = cursor.fetchone()
            if row:
                res = dict(row)
                res["score"] = score
                results.append(res)
                
        conn.close()
        return results

    def search_faiss(self, query, top_k=5):
        """Version 2: FAISS IndexFlatIP dense search."""
        query_vector = EmbeddingService.embed_queries([query]).astype('float32') # Shape: (1, dim)
        
        # Perform query search
        scores, indices = self.faiss_index.search(query_vector, top_k)
        
        results = []
        conn = get_connection()
        cursor = conn.cursor()
        
        # FAISS search returns parallel arrays of scores and indices
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: # Padding value if index is empty
                continue
            cursor.execute("""
                SELECT c.chunk_id, c.document_id, c.parent_chunk_id, c.chunk_type, c.page_number, 
                       c.chapter_title, c.section_title, c.chunk_text, d.document_name, d.circular_number, d.ref_number
                FROM chunks c
                JOIN documents d ON c.document_id = d.document_id
                WHERE c.vector_index = ?
            """, (int(idx),))
            row = cursor.fetchone()
            if row:
                res = dict(row)
                res["score"] = float(score)
                results.append(res)
                
        conn.close()
        return results

if __name__ == "__main__":
    # Test execution
    retriever = DenseRetriever()
    query = "What is the collateral-free loan limit for agricultural credit?"
    print(f"\n--- Testing NumPy Retrieval for: '{query}' ---")
    np_res = retriever.search_numpy(query, top_k=2)
    for r in np_res:
        print(f"Doc: {r['document_name']} (Page {r['page_number']}) | Section: {r['section_title']} | Score: {r['score']:.4f}")
        print(f"Text snippet: {r['chunk_text'][:150]}...\n")
        
    print(f"\n--- Testing FAISS Retrieval for: '{query}' ---")
    faiss_res = retriever.search_faiss(query, top_k=2)
    for r in faiss_res:
        print(f"Doc: {r['document_name']} (Page {r['page_number']}) | Section: {r['section_title']} | Score: {r['score']:.4f}")
        print(f"Text snippet: {r['chunk_text'][:150]}...\n")
