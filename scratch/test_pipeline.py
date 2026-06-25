import os
import sys
import json
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from retrieval.fusion import HybridRetriever
from retrieval.reranker import Reranker
from generation.llm_client import LLMClient
from generation.citation_verifier import CitationVerifier

# Disable HF logs
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

query_text = "What is the collateral-free limit for agricultural credit?"
print("Query:", query_text)

hybrid = HybridRetriever()
candidates = hybrid.search(query_text, top_k=15)
reranked = Reranker.rerank(query_text, candidates, top_k=5)

print("\n--- Raw LLM Input Chunks ---")
for idx, chunk in enumerate(reranked):
    print(f"[{idx+1}] File: {chunk.get('filename')} | Sec: {chunk.get('section_title')}")

print("\n--- Generating Answer ---")
llm_output = LLMClient.generate_answer(query_text, reranked)
print("\n--- Raw LLM Output JSON ---")
print(json.dumps(llm_output, indent=2))

print("\n--- Verifying Citations ---")
final_response = CitationVerifier.verify_citations(llm_output, reranked)
print("\n--- Final Response JSON ---")
print(json.dumps(final_response, indent=2))
