from retrieval.dense import DenseRetriever
from retrieval.sparse import BM25Retriever

class HybridRetriever:
    """Combines BM25 and FAISS dense retrievers using Reciprocal Rank Fusion (RRF)."""

    def __init__(self):
        print("Initializing Hybrid Retriever...")
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = BM25Retriever()
        print("Hybrid Retriever initialized successfully.")

    def search(self, query, top_k=5, rrf_k=60):
        """
        Performs hybrid retrieval using RRF score fusion.
        Retrieves Top 50 candidates from both retrievers and fuses them.
        """
        # Retrieve candidate lists (Top 50 is standard for fusion)
        dense_results = self.dense_retriever.search_faiss(query, top_k=50)
        sparse_results = self.sparse_retriever.search(query, top_k=50)
        
        # Maps chunk_id to fused score and the full chunk record
        fused_scores = {}
        chunk_lookup = {}
        
        # Process dense ranks
        for rank, chunk in enumerate(dense_results):
            chunk_id = chunk["chunk_id"]
            chunk_lookup[chunk_id] = chunk
            # 1-indexed rank
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + (rank + 1)))
            
        # Process sparse ranks
        for rank, chunk in enumerate(sparse_results):
            chunk_id = chunk["chunk_id"]
            chunk_lookup[chunk_id] = chunk
            # 1-indexed rank
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + (rank + 1)))
            
        # Sort chunks by fused score in descending order
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        results = []
        for chunk_id in sorted_ids[:top_k]:
            chunk = chunk_lookup[chunk_id].copy()
            chunk["rrf_score"] = float(fused_scores[chunk_id])
            results.append(chunk)
            
        return results

if __name__ == "__main__":
    # Test execution
    hybrid = HybridRetriever()
    query = "What is the collateral-free agricultural loan limit?"
    print(f"\n--- Testing Hybrid Retrieval for: '{query}' ---")
    results = hybrid.search(query, top_k=3)
    for i, r in enumerate(results):
        # Safely encode/decode to avoid Windows console CP1252 encoding crashes
        clean_text = r['chunk_text'].encode('ascii', 'ignore').decode('ascii')
        clean_doc = r['document_name'].encode('ascii', 'ignore').decode('ascii')
        print(f"Rank {i+1} | Doc: {clean_doc} (Page {r['page_number']}) | RRF Score: {r['rrf_score']:.6f}")
        print(f"Text snippet: {clean_text[:120]}...\n")
