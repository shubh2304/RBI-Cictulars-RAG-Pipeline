import os
import sys
import json
import urllib.request

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from retrieval.fusion import HybridRetriever
from retrieval.reranker import Reranker
from generation.llm_client import LLMClient

# Fetch chunks
query_agri = "What is the collateral-free limit for agricultural credit?"
hybrid = HybridRetriever()
candidates = hybrid.search(query_agri, top_k=15)
reranked = Reranker.rerank(query_agri, candidates, top_k=5)

# Call LLM with simple query and these 5 chunks
query_kcc = "What is the aim of KCC?"
print("Running LLM client with simple KCC query and 5 real retrieved chunks...")
llm_output = LLMClient.generate_answer(query_kcc, reranked)
print("\n--- LLM Output JSON ---")
print(json.dumps(llm_output, indent=2))
