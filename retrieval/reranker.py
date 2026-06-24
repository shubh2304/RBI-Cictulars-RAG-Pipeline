import os
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "BAAI/bge-reranker-base"

class Reranker:
    """Reranks candidate passages using a CrossEncoder model for higher precision."""
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            print(f"Loading reranker model: {RERANKER_MODEL}...")
            # Automatically runs on GPU/CUDA if available, else CPU
            cls._model = CrossEncoder(RERANKER_MODEL)
            print("Reranker model loaded successfully.")
        return cls._model

    @classmethod
    def rerank(cls, query, chunks, top_k=5, filter_noise=True):
        """
        Takes a query and list of chunk dicts, scores them with CrossEncoder,
        and returns the top-K sorted results.
        """
        if not chunks:
            return []
            
        model = cls.get_model()
        
        # Prepare inputs as (query, passage) pairs
        pairs = [(query, c["chunk_text"]) for c in chunks]
        
        # Compute scores (higher scores mean higher relevance)
        scores = model.predict(pairs)
        
        # Zip chunks with their rerank scores
        reranked_chunks = []
        for i, score in enumerate(scores):
            chunk = chunks[i].copy()
            chunk["rerank_score"] = float(score)
            reranked_chunks.append(chunk)
            
        # Sort by rerank score in descending order
        reranked_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        results = reranked_chunks[:top_k]
        
        if filter_noise and results:
            top_score = results[0]["rerank_score"]
            filtered = [results[0]]
            for c in results[1:]:
                # Keep if score is within 0.35 of the top score OR above absolute threshold of 0.1
                if (c["rerank_score"] >= top_score - 0.35) or (c["rerank_score"] >= 0.1):
                    filtered.append(c)
            results = filtered
            
        return results

if __name__ == "__main__":
    # Test execution
    from retrieval.fusion import HybridRetriever
    
    hybrid = HybridRetriever()
    query = "What is the collateral-free agricultural loan limit?"
    
    print("\n--- 1. Fetching Candidates via Hybrid Search ---")
    candidates = hybrid.search(query, top_k=10)
    
    print("\n--- 2. Reranking Candidates via CrossEncoder ---")
    results = Reranker.rerank(query, candidates, top_k=3)
    
    for i, r in enumerate(results):
        clean_text = r['chunk_text'].encode('ascii', 'ignore').decode('ascii')
        clean_doc = r['document_name'].encode('ascii', 'ignore').decode('ascii')
        print(f"Rank {i+1} | Doc: {clean_doc} (Page {r['page_number']}) | Rerank Score: {r['rerank_score']:.4f}")
        print(f"Text snippet: {clean_text[:120]}...\n")
