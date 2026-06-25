import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from retrieval.fusion import HybridRetriever
from retrieval.reranker import Reranker

query = "What is the collateral-free limit for agricultural credit?"
hybrid = HybridRetriever()
candidates = hybrid.search(query, top_k=15)
reranked = Reranker.rerank(query, candidates, top_k=5)

for idx, chunk in enumerate(reranked):
    print(f"=== CHUNK {idx+1} ===")
    print("KEYS:", list(chunk.keys()))
    print("circular_number:", chunk.get("circular_number"))
    print("document_name:", chunk.get("document_name"))
    print("pub_date:", chunk.get("pub_date"))
    print("page_number:", chunk.get("page_number"))
    print("section_title:", chunk.get("section_title"))
    print("source_url:", chunk.get("source_url"))
    print("TEXT:", chunk.get("chunk_text")[:300])
    print("-" * 50)
