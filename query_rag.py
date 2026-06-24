import sys
import json
import os

# Disable Hugging Face progress bars and warning logs in the terminal
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Force transformers to only log errors, avoiding download prints
from transformers.utils import logging as tf_logging
tf_logging.set_verbosity_error()

from retrieval.fusion import HybridRetriever
from retrieval.reranker import Reranker
from generation.llm_client import LLMClient
from generation.citation_verifier import CitationVerifier

def sanitize_text(text):
    """Sanitizes text for safe terminal printing on Windows CP1252/ASCII consoles."""
    if not text:
        return ""
    # Map common unicode characters to clean equivalents
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2010', '-')
    text = text.replace('\u20b9', 'Rs.')
    text = text.replace('\ufffd', '-')
    # Fallback encode/decode to drop any remaining unmappable codes
    return text.encode('ascii', 'replace').decode('ascii')

def query_system(query_text, top_k=5):
    print("=" * 80)
    print(f"Query: {query_text}")
    print("=" * 80)
    
    # 1. Initialize Hybrid Search
    print("[1/4] Running Hybrid Retrieval (BM25 + FAISS)...")
    hybrid = HybridRetriever()
    candidates = hybrid.search(query_text, top_k=15)
    
    if not candidates:
        print("No matches found in the corpus.")
        return
        
    # 2. Rerank Chunks
    print(f"[2/4] Reranking {len(candidates)} candidates using BGE Cross-Encoder...")
    reranked = Reranker.rerank(query_text, candidates, top_k=top_k)
    
    # 3. Generate Answer
    print("[3/4] Generating answer from LLM...")
    llm_output = LLMClient.generate_answer(query_text, reranked)
    
    # 4. Verify Citations
    print("[4/4] Verifying citations against source texts...")
    final_response = CitationVerifier.verify_citations(llm_output, reranked)
    
    # Render final output
    print("\n" + "#" * 80)
    print("                             GENERATED ANSWER                                   ")
    print("#" * 80)
    
    # Print answer
    answer_text = final_response["response"]
    print(sanitize_text(answer_text))
    
    # Print output length metadata
    words = len(answer_text.split())
    chars = len(answer_text)
    print(f"\n[Response Length: {words} words, {chars} characters]")
    
    print("\n" + "-" * 80)
    print("                                CITATIONS                                       ")
    print("-" * 80)
    
    for cit in final_response["citations"]:
        tag = cit["citation_tag"]
        verified = "VERIFIED" if cit["verified"] else "HALLUCINATION WARNING"
        src = cit["source"]
        statement = sanitize_text(cit["statement"])
        doc = sanitize_text(src["document_name"])
        sec = sanitize_text(src["section_title"])
        
        print(f"{tag} [{verified}]")
        print(f"  Statement : \"{statement}\"")
        print(f"  Source    : {doc}")
        print(f"  Location  : Page {src['page_number']} | Section: {sec}")
        if src["ref_number"] or src["circular_number"]:
            print(f"  Details   : Ref: {src['ref_number']} | Circular: {src['circular_number']}")
        print("-" * 50)
        
    if final_response["warnings"]:
        print("\nWarnings:")
        for w in final_response["warnings"]:
            print(f"  - {w}")
            
    print("=" * 80)

if __name__ == "__main__":
    # If a query is provided in the args, run it. Otherwise, prompt the user
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        query_system(user_query)
    else:
        print("Please provide a query as arguments. Example:")
        print("python query_rag.py What is the collateral-free limit for agricultural credit?")
